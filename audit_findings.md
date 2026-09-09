# mandelbrot_julia_py — Audit Findings

## Critical Bugs / Incorrect Behavior

1. **GPU/CPU color mismatch** (`compute_set_gpu` vs `compute_set_cpu`)
   - GPU kernel inlines an *approximate* `overlay()` for color blending (lines 552–563).
   - CPU path calls the exact `overlay()` from `color_pixel()` (lines 348–357).
   - Result: identical mathematical input can produce visibly different pixel colors on GPU vs CPU.

2. **`_blit_surface_clamped` edge fill uses a single corner pixel**
   - Lines 678, 684, 687 sample `(sw-1,0)`, `(0,0)`, `(0,sh-1)` and fill large rectangles with that one color.
   - During panning this creates visible color banding along exposed edges instead of properly rendering the missing strip.

3. **`show-orbits` default is silently contradictory**
   - `DEFAULT_KEYBINDS` includes `"toggle-orbits": "space"`.
   - `run_render_mode` state default is `"show_orbits": True`.
   - `_reset_to_defaults` also sets `"show_orbits": True`.
   - The YAML file has `show-orbits = True`.
   - However, `MenuOverlay._do_action` toggles it via `not state.get("show_orbits", False)` which is fine, but the keybind implies it should default to **off** if the user has never toggled it. Currently there is no distinction between "never touched" and "explicitly on".

4. **`_reset_to_defaults` does not reset `use_gpu`**
   - All other rendering toggles (`smooth`, `fxaa`, `is_julia`) are reset, but `use_gpu` is omitted.
   - After a reset the GPU state persists, which is inconsistent.

## Inefficiencies

5. **`compute_set_cpu` uses `complex()` inside Numba parallel loop**
   - Lines 405–407: `complex(creal[x], cim[y])` constructs a Python complex object per pixel inside `@njit(parallel=True)`.
   - Use separate `zx, zy` floats to avoid the object overhead and enable better vectorization.

6. **`_draw_orbit` rebuilds colortable every frame**
   - Line 790 calls `make_colortable(...)` on every invocation.
   - Cache it alongside the render params cache.

7. **`_render_exposed_edges` rebuilds render params every call**
   - Lines 1239–1244 duplicate the cache lookup from `render_to_surface` instead of receiving params as an argument.
   - Extract a helper or pass params in.

8. **GPU kernel duplicates ~150 lines of shading math**
   - `compute_set_gpu` inlines `smooth_iter`, `blinn_phong`, `color_pixel`, and `overlay`.
   - This is unavoidable for CUDA, but it means any fix to CPU shading must be manually synced to GPU. Consider a shared constants/table approach or generate the kernel from a template.

9. **`apply_fxaa` edge copy is Python-loop-only**
   - Lines 650–663 copy border pixels with plain `for` loops after the Numba parallel region.
   - Inlining or slicing outside Numba would be faster and simpler.

10. **`perf_monitor.py` dead code**
    - `inc_frame()` (line 123) is never called; frame counting is done inline.
    - `find_executable`, `is_nvtop_available`, `get_nvtop_snapshot` are defined but unused anywhere.

11. **`launch_with_monitor.py` hardcodes script name**
    - Line 66 uses `"mandelbrot-testing-playground.py"` instead of deriving it from `__file__` or sys.argv[0].

## Inconsistencies

12. **Naming: snake_case vs kebab-case vs hyphenated YAML keys**
    - Code uses `stripe_s`, `step_s`, `light_angle`, `k_ambiant`, `orbit_max_iter`.
    - YAML uses `stripe_s`, `step_s`, `light_angle`, `k_ambiant`, `orbit-max-iter`.
    - `build_render_params` returns `"ncy"` but callers pass `"ncycle"` in tests.
    - Standardize one convention; recommend snake_case everywhere and a single `state` dict schema.

13. **`color_pixel` smooth vs discrete branches differ slightly from `_orbit_pixel_color`**
    - `color_pixel` smooth: `cniter = math.sqrt(niter) / ncycle` (line 311).
    - `_orbit_pixel_color` smooth: `cniter = math.sqrt(i + 1) / ncycle` (line 755).
    - These are intentionally different (niter is float, i is index), but the duplicated logic should be factored out.

14. **`debug_tools.py` colortable is completely different implementation**
    - Uses HSV angle wrapping and returns raw HSV values instead of RGB.
    - Never imported by the main app. Either delete or align with `make_colortable`.

15. **GPU memory units are inconsistent**
    - `perf_monitor.py` line 49: `gpu_mem_total` converted to **GB** (`/ 1024.0` from MB).
    - Line 100: `gpu_mem_used` left in **MB** (`/ (1024**2)` from bytes).
    - Mixing GB and MB in the same dict (`gpu_mem_used` vs `gpu_mem_total`) causes misleading display in the HUD.

## Minor / Style

16. `iterate_single` allocates `max_iter`-length arrays even for points that escape at iteration 2. Return a list or slice dynamically if memory matters, or document that this is intentional for Numba.

17. `prev_pow2(0)` returns `1`, which is mathematically questionable but harmless. Add a docstring.

18. `MenuOverlay._do_action` has a redundant branch: `elif key == "reset-all"` calls `_reset_to_defaults`, but `_reset_to_defaults` does not reset `use_gpu` (see #4).

19. `test_all.py` line 31 redefines `COLOR_THETAS` after importing it from the module.

20. `test_all.py` line 192 calls `mb.smooth_iter(complex(0.5, 0.5), 100, 0.0, 0.9)` but `smooth_iter` is `@njit` — the `complex` constructor works in object mode but is slower; pass real/imag parts like the GPU kernel does.
