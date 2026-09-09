# Audit & Fix Plan — Mandelbrot Testing Playground

## 1. math.txt Model vs. Code Implementation Audit

### 1.1 Correctly Implemented

| math.txt Concept | Code Location | Status |
|---|---|---|
| Mandelbrot iteration `z_M(n) = z² + c`, z₀=0 | `iterate_single` L142, `compute_orbit` L604, GPU kernel L348 | OK |
| `s_M = (0.018, -0.63)` fixed point | `state["orbit_point"]` L1347, L978, L1636 | OK |
| Escape condition \|z\|² ≤ 4 | `iterate_single` L151 | OK |
| 16 HSV palettes (COLOR_THETAS) | L59-76 | OK |
| Smooth coloring (sigmoid) | `color_pixel` L261 | OK |
| Discrete mode (sqrt fractional) | `color_pixel` L267 | OK |
| Blinn-Phong lighting (4 params) | `blinn_phong` L214, GPU L409 | OK |
| GPU normal vector fix | GPU kernel L382: `normal_re = zr*dzr + zi*dzi` | OK |
| Orbit trajectory computation | `compute_orbit` L604 | OK |
| Liang-Barsky clipping | `_liang_barsky_clip` L635 | OK |
| FXAA anti-aliasing | `apply_fxaa` L523 | OK |
| Persistent YAML settings | `load_settings` L82, `save_settings` L100 | OK |
| Phase shifting | `make_colortable` L187 | OK |
| Stripe effects | `smooth_iter` L231 | OK |
| Power-of-2 iter stepping | L1565-1581 (`iter-up`/`iter-down`) | OK |

### 1.2 NOT Implemented from math.txt

- **Julia set (`z_J`)**: `z_J(0,x,y) = (x,y)`, `z_J(n) = z² + c_J` with `c_J = (0.394, 0.338)`, `s_J = (0.153, 0.473)` — absent entirely. Only Mandelbrot rendering exists.
- **Periodic cycle detection**: `C_MEL`, `C_Mc`, `C_ML` sliding-window analysis — absent.
- **Fixed point computation**: `Z_Mreal`, `Z_Mimag`, `Z_Mpos`, `Z_Mneg` quadratic formula — absent.
- **Type classification**: `T_Ms`/`T_Js` (Periodic/Siegel/Collapsing) — absent.
- **Audio mapping**: `tone(...)` from orbit values — absent.
- **`s_J` (Julia fixed point)**: Not in code. Only `s_M` is used as orbit_point default.
- **Siegel disk detection**: `A_s` accuracy threshold and `A_sp` slope threshold — absent.

### 1.3 Mathematical Inconsistencies

#### 1.3.1 Escape Radius Discrepancy
- `iterate_single` (L151): uses `zx² + zy² > 4` (escape radius = 2)
- `smooth_iter` (L233): uses `esc_radius_2 = 10.0 ** 10` (escape radius = 100,000)
- With `max_iter = 128` (typical), a point escaping at radius 2 would need ~17 iterations, but with radius 100K it needs ~34+ iterations. Many points that should be exterior will hit max_iter and render as interior (black).
- **Impact**: When `step_s > 0` or stripe effects are active, GPU and CPU diverge significantly from the math.txt model.
- **Fix**: Change `esc_radius_2` to 4.0 (or make it a parameter). The smooth coloring formula already uses `math.log(esc_radius_2)` so it adapts.

#### 1.3.2 `step_s` Formula Divergence (CPU vs GPU)
- **CPU** (`color_pixel` L293): `light_step = 6 * (1 - x**5 - (1-x)**100) / 10`
- **GPU** (kernel L441): `light_step = 6 * (1 - math.pow(x2, 5) - math.pow(1 - x2, 30)) / 10`
- CPU uses exponent **100**, GPU uses exponent **30**.
- **Impact**: Step effect renders differently on CPU vs GPU. This is the root cause of the "max 137 from 0 boundary pixels" difference when step_s is active — the user noted this but attributed it to floating point.
- **Fix**: Unify on exponent 100 (or 30) in both paths.

#### 1.3.3 `normal` Zero-Division (CPU only)
- `blinn_phong` (L215): `normal = normal / abs(normal)` — no guard for `abs(normal) == 0`
- GPU kernel (L411-427): guards with `if mag > 0:`, sets `bright = 0.0` otherwise
- **Impact**: If `z/dz` evaluates to 0+0j (e.g., c = 0+0j), CPU produces NaN; GPU produces 0.0
- **Fix**: Add guard in `blinn_phong` or in `color_pixel` before calling it.

### 1.4 Dead Code / Unused Params

- `_orbit_pixel_color` (L661): parameter `n` is never used in the function body (L661-681)
- `col_i_s` computed in both CPU and GPU but never used for color lookup (only `col_i` is used)

---

## 2. Test File Fixes (`test_all.py`)

### 2.1 Import Mechanism
- **Problem**: Line 14 uses `from mandelbrot_testing_playground import ...` — hyphens in filename make this fail.
- **Fix**: Use `importlib.util.spec_from_file_location('mb', 'mandelbrot-testing-playground.py')` to load the module.

### 2.2 `iterate_single` Return Value Mismatch
- **Problem**: `iterate_single` returns `(orbit_x, orbit_y, escaped)` — a 3-tuple (L152-153)
- **Test** (L146-151): `orbit_ext, n = iterate_single(0.5, 0.5, 50)` — unpacks 2 values from 3-tuple → `ValueError: too many values to unpack`
- **Fix**: Change to `orbit_ext_x, orbit_ext_y, escaped = iterate_single(...)` and use `len(orbit_ext_x)` for iteration count.

### 2.3 `compute_orbit` Length Assertion
- **Problem**: `compute_orbit(-0.75, 0.0, 50)` returns array of shape `(51, 2)` when bounded (initial point + 50 iterations), but test (L137) expects `len(orbit) == 50`
- **Fix**: Change assertion to `len(orbit) <= 51` or `len(orbit) == 51`

### 2.4 Missing Tests (to add)
- Palette coverage: verify all 16 COLOR_THETAS produce valid colortables
- Colortable hue wrapping: verify phase shift changes hue correctly
- CPU/GPU consistency (skip if CUDA unavailable)
- `build_render_params` correctness for all fields
- `_do_action` behavior: cyclic wrapping, step multipliers, power-of-2 max_iter
- MenuOverlay click handling: minus/plus/reset button unpacking
- FXAA: verify output shape, edge handling
- Strip effects: verify stripe_s produces different output than stripe_s=0
- `step_s` consistency between CPU and GPU (this is the test that would catch the exponent bug)
- Settings persistence round-trip (save then load)
- `next_pow2`/`prev_pow2` correctness

---

## 3. Module-Level `argparse` Side Effect
- **Problem**: `parser.parse_args()` runs at import time (L1816). Importing the module for tests triggers argparse on `sys.argv`.
- **Fix**: Move argparse block inside `if __name__ == "__main__":` block.

---

## 4. MenuOverlay `_do_action` Issues

### 4.1 `max_iter` Ctrl Modifier Non-Functional
- **Problem**: Ctrl modifier for max_iter (L921-923): `delta = 1 * 0.1 = 0.1`, then `new_log2 = int(math.floor(log2_val + 0.1))`. Since `log2_val` is integer (for power-of-2 values), `floor(log2_val + 0.1) == log2_val` → no change.
- **Fix**: The power-of-2 stepping should handle fractional log2 changes differently, or ctrl should step by 1 (not 0.1) in log2 space.

### 4.2 Inconsistent Cyclic Wrapping (Keyboard vs Click)
- **Keyboard** (`step_val`, L1272-1274): clamps to [lo, hi] — no wrapping for cyclic params
- **Click** (`_do_action`, L939-942): wraps cyclic params with modulo
- **Impact**: e.g., pressing `ctrl+r` to increase hue_0 past 1.0 will clamp at 1.0; clicking the menu + button will wrap to 0.0.
- **Fix**: Make `step_val` wrap for cyclic params, or add a `cyclic` parameter.

### 4.3 Inconsistent `stripe-down` Behavior
- **Keyboard**: `stripe-up` wraps with `% 101` (L1524), but `stripe-down` clamps with `max(0.0, ...)` (L1525)
- **Fix**: Make both wrap, or both clamp.

### 4.4 `orbit_drag` Does Not Trigger Full Re-render
- **Problem**: L1467-1475: orbit drag sets `needs_render = True` but the render path at L1662 only does an offset blit (reusing cached surface) or full re-render. Since orbit point changed but `rgb_thetas`/`phase`/`max_iter` haven't, the cache key is the same, so the re-rendered surface uses the OLD orbit point. The orbit is drawn on top (L1719 `_draw_orbit`), but the underlying surface doesn't change.
- **Issue**: This is actually by design — orbit is drawn as an overlay, not part of the surface. The orbit point drag correctly triggers a redraw, and `_draw_orbit` uses the updated `state["orbit_point"]`. **Status**: OK — orbit is drawn as overlay, not baked into the surface.

---

## 5. Execution Order — ALL COMPLETE

### Phase 1: Test Suite Rewrite (highest priority) ✅
1. Fix `test_all.py` import mechanism using `importlib`
2. Fix `iterate_single` and `compute_orbit` test assertions
3. Add comprehensive new tests covering: palettes, colortable, orbit, CPU/GPU consistency, keybind actions, settings persistence, MenuOverlay, FXAA, stripe effects, step_s consistency, next_pow2/prev_pow2, build_render_params
4. Run tests, fix failures

### Phase 2: Bug Fixes ✅
5. Fix `step_s` exponent divergence (100 vs 30) — unified to 100 in GPU kernel
6. Fix `esc_radius_2` in `smooth_iter` and GPU kernel (10^10 → 4.0)
7. Add zero-normal guard in `blinn_phong`
8. Move `argparse.parse_args()` inside `if __name__ == "__main__":`
9. Fix `next_pow2`/`prev_pow2` bit_length off-by-one bugs

### Phase 3: MenuOverlay Consistency ✅
10. Fix `max_iter` ctrl modifier in `_do_action` — removed `* 0.1` that made it a no-op
11. Added `step_val_cyclic` for keyboard cyclic params — wraps instead of clamps
12. Fix `stripe-down` to wrap consistently with `stripe-up` (use `% 101`)
13. Change `light_angle` and `light_azim` step from 0.02 to 0.1 (36° per keypress)

### Phase 4: Verification ✅
14. Run `python -m py_compile` on all modules — ALL PASS
15. Run full test suite — 190 passed, 0 failed
16. Verify CPU/GPU consistency — tested in test suite (avg diff small, max diff bounded)

## 6. Additional Bugs Found During Menu Testing

### 6.1 Keybind Collision: `toggle-info` (m) overwritten by `sat-down` (ctrl+m)
- **Problem**: `key_map` is a flat dict mapping `key_code → (action, mod_val)`. Both `toggle-info` and `sat-down` map to `K_m`. The second binding (`sat-down` with `KMOD_CTRL`) overwrites the first (`toggle-info` with mod=0). When 'm' is pressed without ctrl, the lookup finds `("sat-down", KMOD_CTRL)`, sees ctrl not held, and skips — `toggle-info` never fires, so the menu can't be opened.
- **Fix**: Changed `key_map` to `key_code → list of (action, mod_val)` using `setdefault`, and changed the KEYDOWN lookup to iterate bindings, preferring an exact modifier match over a no-modifier fallback.
- **Location**: `mandelbrot_testing_playground.py` L1294-1303 (key_map construction), L1522-1539 (KEYDOWN handler)
