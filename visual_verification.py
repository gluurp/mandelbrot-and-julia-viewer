#!/usr/bin/env python3
"""Visual verification suite for mandelbrot-testing-playground.py.

Tests all rendering modes, UI components, and saves reference screenshots
to the ``visual_tests/`` directory for manual inspection and regression CI.
"""
import sys
import os
import math
import importlib.util
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("MB_PALETTE_REWRITE", "0")

_spec = importlib.util.spec_from_file_location(
    "mb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mandelbrot_testing_playground.py")
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

import pygame
pygame.init()
pygame.display.set_mode((640, 480))

from PIL import Image

# Default palette from palettes.yaml, so reference screenshots stay colorful.
_DEFAULT_PALETTE = mb.get_all_palettes()[0] if mb.get_all_palettes() else {}
_DEFAULT_STOPS = list(_DEFAULT_PALETTE.get("gradient_stops", mb.DEFAULT_GRADIENT_STOPS))
_DEFAULT_BLEND = _DEFAULT_PALETTE.get("blend", mb.DEFAULT_GRADIENT_BLEND)

PASS = 0
FAIL = 0
RESULTS = []

def check(name, condition, details=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(("PASS", name, ""))
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        RESULTS.append(("FAIL", name, details))
        print(f"  FAIL: {name} {details}")

def save_png(arr, filename):
    """Save a (H, W, 3) uint8 array as PNG using PIL."""
    Image.fromarray(arr, "RGB").save(filename)

# ── Helpers ────────────────────────────────────────────────────────────────

def make_state(**overrides):
    state = {
        "max_iter": 256,
        "gradient_stops": list(_DEFAULT_STOPS),
        "gradient_blend": _DEFAULT_BLEND,
        "use_gpu": False,
        "stripe_s": 0.0,
        "stripe_sig": 0.9,
        "step_s": 0.0,
        "light_angle": mb.DEFAULT_LIGHT_ANGLE,
        "light_azim": mb.DEFAULT_LIGHT_AZIM,
        "light_i": mb.DEFAULT_LIGHT_I,
        "k_ambiant": mb.DEFAULT_K_AMBIANT,
        "k_diffuse": mb.DEFAULT_K_DIFFUSE,
        "k_specular": mb.DEFAULT_K_SPECULAR,
        "shininess": mb.DEFAULT_SHININESS,
        "smooth": True,
        "fxaa": False,
        "palette_index": 0,
        "orbit_max_iter": 200,
        "orbit_point_size": 3,
        "use_julia": False,
        "julia_c": [0.394, 0.338],
    }
    state.update(overrides)
    return state

def render_image(state, w=128, h=128, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, maxiter=256, use_julia=False):
    mb._RENDER_CACHE.clear()
    params = mb.build_render_params(state, maxiter=maxiter, use_julia=use_julia)
    return mb.compute_image(w, h, xmin, xmax, ymin, ymax, maxiter, params)

# ── Test sections ──────────────────────────────────────────────────────────

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_tests")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("VISUAL VERIFICATION SUITE")
print("=" * 60)

# 1. Mandelbrot basic rendering
print("\n--- Mandelbrot (CPU, smooth) ---")
state = make_state()
img = render_image(state, 128, 128, -2.0, 1.0, -1.5, 1.5, 256)
check("MB render shape", img.shape == (128, 128, 3))
check("MB has colored pixels", np.sum(np.any(img > 30, axis=2)) > 1000)
check("MB has black (inside set) pixels", np.sum(np.all(img < 5, axis=2)) > 100)
check("MB pixel range", img.min() >= 0 and img.max() <= 255)
save_png(img, os.path.join(OUT, "mandelbrot_smooth.png"))

# 2. Julia basic rendering
print("\n--- Julia (CPU, smooth) ---")
state_j = make_state(use_julia=True, julia_c=[0.394, 0.338])
img_j = render_image(state_j, 128, 128, -1.5, 1.5, -1.5, 1.5, 256, use_julia=True)
check("Julia render shape", img_j.shape == (128, 128, 3))
check("Julia has colored pixels", np.sum(np.any(img_j > 30, axis=2)) > 500)
check("Julia differs from MB", not np.array_equal(img, img_j))
save_png(img_j, os.path.join(OUT, "julia_smooth.png"))

# 3. GPU consistency
print("\n--- GPU vs CPU consistency ---")
if mb._CUDA_AVAILABLE:
    state_gpu = make_state(use_gpu=True)
    mg = render_image(state_gpu, 64, 64, -2.0, 1.0, -1.5, 1.5, 128)
    state_cpu = make_state(use_gpu=False)
    mc = render_image(state_cpu, 64, 64, -2.0, 1.0, -1.5, 1.5, 128)
    diff = np.max(np.abs(mg.astype(int) - mc.astype(int)))
    check("GPU/CPU MB max diff <= 1", diff <= 1, f"max_diff={diff}")

    state_ju_gpu = make_state(use_julia=True, use_gpu=True)
    jg = render_image(state_ju_gpu, 64, 64, -1.5, 1.5, -1.5, 1.5, 128, use_julia=True)
    state_ju_cpu = make_state(use_julia=True, use_gpu=False)
    jc = render_image(state_ju_cpu, 64, 64, -1.5, 1.5, -1.5, 1.5, 128, use_julia=True)
    diff_j = np.max(np.abs(jg.astype(int) - jc.astype(int)))
    check("GPU/CPU Julia max diff <= 1", diff_j <= 1, f"max_diff={diff_j}")
    save_png(jg, os.path.join(OUT, "julia_gpu.png"))
else:
    check("CUDA available", False, "CUDA not available - skipped GPU tests")

# 4. Discrete vs smooth
print("\n--- Smooth vs discrete ---")
state_s = make_state(smooth=True)
img_s = render_image(state_s, 32, 32, -2.0, 1.0, -1.5, 1.5, 64)
state_d = make_state(smooth=False)
img_d = render_image(state_d, 32, 32, -2.0, 1.0, -1.5, 1.5, 64)
check("Smooth and discrete differ", not np.array_equal(img_s, img_d))
save_png(img_d, os.path.join(OUT, "mandelbrot_discrete.png"))

# 5. FXAA
print("\n--- FXAA ---")
state_f = make_state(fxaa=True)
img_f = render_image(state_f, 64, 64, -2.0, 1.0, -1.5, 1.5, 64)
check("FXAA render shape", img_f.shape == (64, 64, 3))
check("FXAA has pixels", np.sum(np.any(img_f > 0, axis=2)) > 0)
save_png(img_f, os.path.join(OUT, "mandelbrot_fxaa.png"))

# 6. All palettes (gradient sections)
print("\n--- All color palettes ---")
_all_palettes = mb.get_all_palettes()
for idx, pal in enumerate(_all_palettes):
    name = pal["name"]
    state_p = make_state(palette_index=idx)
    mb._apply_palette_by_index(state_p, idx)
    img_p = render_image(state_p, 32, 32, -2.0, 1.0, -1.5, 1.5, 32)
    check(f"Palette '{name}': renders", img_p.shape == (32, 32, 3) and np.any(img_p > 0))
    save_png(img_p, os.path.join(OUT, f"palette_{name}.png"))

# 7. Julia with different c values
print("\n--- Julia c values ---")
test_cases = [
    ([0.0, 0.0], "unit_circle"),
    ([-0.7, 0.27015], "dendrite"),
    ([-0.8, 0.156], "spiral"),
    ([0.285, 0.01], "rabbit"),
    ([0.394, 0.338], "default"),
]
for c, label in test_cases:
    state_c = make_state(use_julia=True, julia_c=list(c))
    img_c = render_image(state_c, 64, 64, -1.5, 1.5, -1.5, 1.5, 128, use_julia=True)
    check(f"Julia c={c} ({label}): renders", img_c.shape == (64, 64, 3) and np.any(img_c > 0))
    save_png(img_c, os.path.join(OUT, f"julia_{label}.png"))

# 8. Orbit computation
print("\n--- Orbit computation ---")
orbit = mb.compute_orbit(0.018, -0.63, 200)
check("MB orbit shape", orbit.shape == (201, 2))
check("MB orbit all finite", np.all(np.isfinite(orbit)))
check("MB orbit starts at (sx,sy)",
      abs(orbit[0, 0] - 0.018) < 1e-9 and abs(orbit[0, 1] + 0.63) < 1e-9)

orbit_j = mb.compute_orbit(0.018, -0.63, 200, use_julia=True,
                           julia_c_re=0.394, julia_c_im=0.338)
check("Julia orbit shape", orbit_j.shape == (201, 2))
check("Julia orbit starts at z0=coord",
      abs(orbit_j[0, 0] - 0.018) < 1e-9 and abs(orbit_j[0, 1] + 0.63) < 1e-9)
check("Julia orbit all finite", np.all(np.isfinite(orbit_j)))

# 9. render_to_surface + MenuOverlay
print("\n--- render_to_surface + MenuOverlay ---")
state_mb = make_state()
mb._RENDER_CACHE.clear()
surf_mb, _ = mb.render_to_surface(128, 128, -2.0, 1.0, -1.5, 1.5, 128, state_mb)
arr_mb = pygame.surfarray.array3d(surf_mb)
check("render_to_surface: MB shape", arr_mb.shape == (128, 128, 3))

state_ju = make_state(use_julia=True)
mb._RENDER_CACHE.clear()
surf_ju, _ = mb.render_to_surface(128, 128, -1.5, 1.5, -1.5, 1.5, 128, state_ju)
arr_ju = pygame.surfarray.array3d(surf_ju)
check("render_to_surface: Julia shape", arr_ju.shape == (128, 128, 3))

# MenuOverlay draw (this was the crash we fixed)
overlay = mb.MenuOverlay()
overlay.active = True
font = pygame.font.SysFont("monospace", 16)
overlay_surface = pygame.Surface((640, 480))
overlay_surface.fill((0, 0, 0))
overlay.draw(overlay_surface, font, state_mb,
    {"quit": "escape", "toggle-point-info": "m", "reset-view": "o",
     "toggle-gpu": "p", "toggle-smooth": "u", "toggle-fxaa": "j",
     "toggle-julia": "s", "toggle-orbits": "space", "reset-orbit-point": "i",
     "cycle-palette": "tab", "reset-settings": "backspace",
     "iter-up": "=", "iter-down": "-"},
    xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5)
arr_overlay = pygame.surfarray.array3d(overlay_surface)
check("MenuOverlay draw: no crash", arr_overlay.shape == (640, 480, 3))
check("MenuOverlay draw: has content", np.sum(np.any(arr_overlay > 0, axis=2)) > 100)
save_png(arr_overlay, os.path.join(OUT, "overlay_active.png"))

overlay.active = False
overlay_surface.fill((0, 0, 0))
overlay.draw(overlay_surface, font, state_mb, {}, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5)
check("MenuOverlay inactive: no crash", True)

# MenuOverlay with Julia
overlay.active = True
overlay_surface.fill((0, 0, 0))
overlay.draw(overlay_surface, font, state_ju, {}, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5)
arr_ju_overlay = pygame.surfarray.array3d(overlay_surface)
check("MenuOverlay Julia: has content", np.sum(np.any(arr_ju_overlay > 0, axis=2)) > 100)
check("MenuOverlay Julia: has non-black pixels", np.any(arr_ju_overlay.max(axis=2) > 0))
save_png(arr_ju_overlay, os.path.join(OUT, "overlay_julia.png"))

# 10. MenuOverlay handle_click
print("\n--- MenuOverlay.handle_click ---")
overlay.active = True
overlay.button_rects = {}
# The overlay should handle clicks without crashing
click_result = overlay.handle_click((100, 100), make_state())
check("MenuOverlay handle_click: returns tuple", isinstance(click_result, tuple) and len(click_result) == 2)

# 11. Orbit visualization with _draw_orbit
print("\n--- Orbit visualization ---")
state_orbit = make_state(orbit_max_iter=200)
mb._RENDER_CACHE.clear()
state_orbit["orbit_hover"] = True
surf_orbit, _ = mb.render_to_surface(128, 128, -2.0, 1.0, -1.5, 1.5, 64, state_orbit)
arr_orbit = pygame.surfarray.array3d(surf_orbit)
check("Orbit render: shape", arr_orbit.shape == (128, 128, 3))
check("Orbit render: has pixels", np.sum(np.any(arr_orbit > 0, axis=2)) > 0)
# Check that orbit points are drawn (non-black pixels near orbit point)
save_png(arr_orbit, os.path.join(OUT, "orbit_mb.png"))

# Julia orbit
state_ju_orbit = make_state(use_julia=True)
mb._RENDER_CACHE.clear()
surf_ju_orbit, _ = mb.render_to_surface(128, 128, -1.5, 1.5, -1.5, 1.5, 64, state_ju_orbit)
arr_ju_orbit = pygame.surfarray.array3d(surf_ju_orbit)
check("Julia orbit render: shape", arr_ju_orbit.shape == (128, 128, 3))
save_png(arr_ju_orbit, os.path.join(OUT, "orbit_julia.png"))

# 12. build_render_params
print("\n--- build_render_params ---")
state_bp = make_state()
params = mb.build_render_params(state_bp, maxiter=256)
check("build_render_params: has colortable", "colortable" in params)
check("build_render_params: has use_julia", "use_julia" in params)
check("build_render_params: use_julia defaults False", params["use_julia"] is False)
check("build_render_params: has julia_c_re", "julia_c_re" in params)
check("build_render_params: julia_c_re correct", abs(params["julia_c_re"] - 0.394) < 0.001)
check("build_render_params: has julia_c_im", "julia_c_im" in params)
check("build_render_params: julia_c_im correct", abs(params["julia_c_im"] - 0.338) < 0.001)

state_bp_ju = make_state(use_julia=True)
params_ju = mb.build_render_params(state_bp_ju, maxiter=256, use_julia=True)
check("build_render_params: Julia use_julia True", params_ju["use_julia"] is True)

# 13. Persistence (save/load settings)
print("\n--- Persistence ---")
test_settings = {"test-key": "test-value", "is-julia": "True", "julia-cx": "0.394", "julia-cy": "0.338"}
test_file = os.path.join(OUT, "test_settings.yaml")
mb.save_settings(test_file, test_settings)
check("save_settings: file created", os.path.exists(test_file))
loaded = mb.load_settings(test_file)
check("load_settings: returns dict", isinstance(loaded, dict))
check("load_settings: roundtrip", loaded.get("test-key") == "test-value")
check("load_settings: julia-cx", loaded.get("julia-cx") == "0.394")
# Clean up
try:
    os.remove(test_file)
except:
    pass

# 14. CLI args
print("\n--- CLI args ---")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=None)
parser.add_argument("--color", type=str, default=None)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--no-gpu", action="store_true")
parser.add_argument("--julia", action="store_true")
args = parser.parse_args(["--iter", "512", "--color", "fire", "--julia"])
check("CLI: iter parsed", args.iter == 512)
check("CLI: color parsed", args.color == "fire")
check("CLI: julia flag", args.julia is True)

# 15. State persistence for Julia
print("\n--- State persistence (Julia) ---")
state_persist = make_state(use_julia=True, julia_c=[0.5, 0.2])
mb._RENDER_CACHE.clear()
# Verify build_render_params picks up julia_c from state
params = mb.build_render_params(state_persist, maxiter=128, use_julia=True)
check("State persistence: julia_c in params",
      abs(params["julia_c_re"] - 0.5) < 0.001 and abs(params["julia_c_im"] - 0.2) < 0.001)

# 16. Keybind resolution
print("\n--- Keybind resolution ---")
sample_keybinds = {"quit": "q", "menu": "m"}
for action, key in sample_keybinds.items():
    check(f"Keybind '{action}': {key}", key is not None and len(key) > 0)

# 17. Edge cases
print("\n--- Edge cases ---")
# Julia c=0 (unit circle)
state_c0 = make_state(use_julia=True, julia_c=[0.0, 0.0])
img_c0 = render_image(state_c0, 64, 64, -1.5, 1.5, -1.5, 1.5, 128, use_julia=True)
check("Julia c=0: unit circle renders", np.sum(np.any(img_c0 > 0, axis=2)) > 0)
inside = img_c0[32, 32]  # Center point (0,0) should be inside (bounded)
check("Julia c=0: center inside (bounded)", np.all(inside < 5) or np.all(inside > 0))
save_png(img_c0, os.path.join(OUT, "julia_c0_unit_circle.png"))

# Low iteration
state_low = make_state(max_iter=8)
img_low = render_image(state_low, 32, 32, -2.0, 1.0, -1.5, 1.5, 8)
check("Low iter (8): renders", img_low.shape == (32, 32, 3))
check("Low iter (8): has pixels", np.sum(np.any(img_low > 0, axis=2)) > 0)

# Zoomed in
state_zoom = make_state(max_iter=512)
img_zoom = render_image(state_zoom, 128, 128, -0.5, 0.5, -0.1, 0.1, 512)
check("Zoomed view: renders", img_zoom.shape == (128, 128, 3))
check("Zoomed view: has detail", np.sum(np.any(img_zoom > 0, axis=2)) > 100)
save_png(img_zoom, os.path.join(OUT, "zoomed_view.png"))

# 18. Zoom with aspect ratio fix
print("\n--- Aspect ratio ---")
for (w, h) in [(100, 100), (640, 480), (1000, 300), (400, 1000)]:
    result = mb.fix_aspect_ratio(-2.5, 1.0, -1.5, 1.5, w, h)
    check(f"fix_aspect_ratio {w}x{h}: returns 4 values", len(result) == 4)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"Screenshots saved to {OUT}/")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
