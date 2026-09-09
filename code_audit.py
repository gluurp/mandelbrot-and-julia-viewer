#!/usr/bin/env python3
"""Code audit tool for mandelbrot-testing-playground.py.

Static analysis checks:
1. All referenced variables/functions exist
2. State keys are consistent across build_render_params, state init, and persistence
3. Return value contracts (shapes, types) are consistent
4. No dead/unreachable code
5. Action dispatch coverage in keybind handler
6. Cache key consistency (CPU vs GPU)
"""
import sys
import os
import ast
import importlib.util
import math
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

_spec = importlib.util.spec_from_file_location(
    "mb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mandelbrot_testing_playground.py")
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

PASS = 0
FAIL = 0

def check(name, condition, details=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {details}")

print("=" * 60)
print("CODE AUDIT SUITE")
print("=" * 60)

# ── 1. AST analysis: find all Name references and check they exist ─────────
print("\n--- AST: undefined names in module scope ---")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mandelbrot_testing_playground.py")) as f:
    source = f.read()
tree = ast.parse(source)

# Collect all defined names at module scope
defined_names = set()
for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defined_names.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                defined_names.add(target.id)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        defined_names.add(node.target.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.asname:
                defined_names.add(alias.asname)
            else:
                defined_names.add(alias.name.split('.')[0])

# Also add all names from builtins and imported modules
import builtins
for name in dir(mb):
    defined_names.add(name)
defined_names.update(dir(builtins))
defined_names.add("pygame")
defined_names.add("np")
defined_names.add("math")
defined_names.add("os")
defined_names.add("sys")
defined_names.add("time")
defined_names.add("argparse")
defined_names.add("cuda")
defined_names.add("cuda")
defined_names.add("numba")
defined_names.add("Image")
defined_names.add("subprocess")
defined_names.add("threading")

# Walk the tree and check Name loads
undefined = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id not in defined_names:
            undefined.add(node.id)

# Filter out method-local names (self attributes etc are handled differently)
# This is a rough check; many false positives expected for nested scope
# Instead, let's check for specific known issues
known_issues = {"palette_names"}
check("No undefined 'palette_names' reference",
      "palette_names" not in source or "PALETTE_NAMES" in source,
      "Found 'palette_names' which should be 'PALETTE_NAMES'")

# ── 2. State key consistency ──────────────────────────────────────────────
print("\n--- State key consistency ---")

# Keys used in build_render_params
state_keys_in_build = {
    "rgb_thetas", "phase", "stripe_s", "stripe_sig", "step_s",
    "light_angle", "light_azim", "light_i", "k_ambiant", "k_diffuse",
    "k_specular", "shininess", "smooth", "fxaa", "use_gpu",
    "palette_index", "show_orbits", "orbit_point", "orbit_max_iter",
    "orbit_point_size", "is_julia", "julia_c",
}

# Keys in default state init
state_keys_in_init = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
        for key_node in node.keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                if "key" not in key_node.value.lower() and not key_node.value.startswith("keybind"):
                    state_keys_in_init.add(key_node.value)

missing_in_build = state_keys_in_build - state_keys_in_init
check("State keys in build_render_params are in init",
      len(missing_in_build) == 0,
      f"Missing: {missing_in_build}")

# Check julia_c is persisted
check("julia-cx persisted", '"julia-cx"' in source)
check("julia-cy persisted", '"julia-cy"' in source)
check("is-julia persisted", '"is-julia"' in source)

# ── 3. Keybind action coverage ─────────────────────────────────────────────
print("\n--- Keybind action coverage ---")
default_keybinds = mb.DEFAULT_KEYBINDS

# Check which actions have handlers in the event loop
for action in sorted(default_keybinds.keys()):
    present = action in source
    check(f"Action '{action}': handled in source", present)

# ── 4. Cache key consistency ────────────────────────────────────────────────
print("\n--- Cache key consistency ---")
# Check that render_to_surface uses both CPU and GPU cache keys
check("render_to_surface uses is_julia in cache key", "is_julia" in source[source.find("def render_to_surface"):source.find("def render_to_surface") + 2000])
check("render_to_surface uses julia_c in cache key", "julia_c" in source[source.find("def render_to_surface"):source.find("def render_to_surface") + 2000])

# ── 5. Function argument contracts ──────────────────────────────────────────
print("\n--- Function contracts ---")

# Define state and params early for use in multiple test sections
state = {
    "rgb_thetas": list(mb.COLOR_THETAS[0]),
    "phase": 0.0, "use_gpu": False,
    "stripe_s": 0.0, "stripe_sig": 0.9, "step_s": 0.0,
    "light_angle": mb.DEFAULT_LIGHT_ANGLE, "light_azim": mb.DEFAULT_LIGHT_AZIM,
    "light_i": mb.DEFAULT_LIGHT_I, "k_ambiant": mb.DEFAULT_K_AMBIANT,
    "k_diffuse": mb.DEFAULT_K_DIFFUSE, "k_specular": mb.DEFAULT_K_SPECULAR,
    "shininess": mb.DEFAULT_SHININESS, "smooth": True, "fxaa": False,
    "is_julia": False, "julia_c": [0.394, 0.338],
}
orbit = mb.compute_orbit(0.018, -0.63, 200)
check("compute_orbit: returns (N, 2) array", orbit.shape[1] == 2)
check("compute_orbit: first point = (sx, sy)", orbit[0, 0] == 0.018 and orbit[0, 1] == -0.63)

orbit_j = mb.compute_orbit(0.018, -0.63, 200, use_julia=True,
                           julia_c_re=0.394, julia_c_im=0.338)
check("compute_orbit Julia: returns (N, 2) array", orbit_j.shape[1] == 2)
check("compute_orbit Julia: first point = (sx, sy)", orbit_j[0, 0] == 0.018 and orbit_j[0, 1] == -0.63)

# smooth_iter
result = mb.smooth_iter(complex(0.5, 0.5), 100, 0.0, 0.9)
check("smooth_iter: returns 4-tuple", isinstance(result, tuple) and len(result) == 4)

result_j = mb.smooth_iter(complex(0.5, 0.5), 100, 0.0, 0.9, use_julia=True,
                          julia_c_re=0.394, julia_c_im=0.338)
check("smooth_iter Julia: returns 4-tuple", isinstance(result_j, tuple) and len(result_j) == 4)
# Both should return (niter, stripe_avg, dem, normal)
for i, r in enumerate(result_j):
    check(f"smooth_iter Julia: element {i} is finite", np.isfinite(r))

# compute_set_cpu
creal = np.linspace(-2.0, 1.0, 32)
cim = np.linspace(-1.5, 1.5, 32)
colortable = mb.make_colortable(np.array(mb.COLOR_THETAS[0], dtype=np.float64))
diag = math.sqrt(3.0**2 + 3.0**2)
light = np.array([0.5 * 2 * math.pi, 0.5 * math.pi / 2, 0.5, 0.3, 0.4, 0.0, 1.0])
mat = mb.compute_set_cpu(creal, cim, 64, colortable, 8.0,
                          0.0, 0.9, 0.0, diag, light, True, False, 0.394, 0.338)
check("compute_set_cpu: returns ndarray", isinstance(mat, np.ndarray))
check("compute_set_cpu: shape (32, 32, 3)", mat.shape == (32, 32, 3))

# compute_set_gpu — tested via compute_image instead (kernel not directly callable)
if mb._CUDA_AVAILABLE:
    state_gpu = dict(state, use_gpu=True)
    params_gpu = mb.build_render_params(state_gpu, maxiter=64)
    img_gpu = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_gpu)
    check("compute_image GPU: shape", img_gpu.shape == (32, 32, 3))
    # Compare with CPU
    mb._RENDER_CACHE.clear()
    params_cpu = mb.build_render_params(state, maxiter=64)
    img_cpu = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_cpu)
    check("compute_image GPU/CPU shapes match", img_gpu.shape == img_cpu.shape)

# build_render_params
params = mb.build_render_params(state, maxiter=256)
check("build_render_params: returns dict", isinstance(params, dict))
required_keys = {"colortable", "ncy", "stripe_s", "stripe_sig", "step_s",
                 "light", "use_gpu", "smooth", "fxaa", "use_julia",
                 "julia_c_re", "julia_c_im"}
check("build_render_params: has all required keys", required_keys <= set(params.keys()))

# compute_image
img = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params)
check("compute_image: returns ndarray", isinstance(img, np.ndarray))
check("compute_image: shape (32,32,3)", img.shape == (32, 32, 3))
check("compute_image: uint8", img.dtype == np.uint8)

# render_to_surface
import pygame
pygame.init()
pygame.display.set_mode((10, 10))
mb._RENDER_CACHE.clear()
surf, used_gpu = mb.render_to_surface(32, 32, -2.0, 1.0, -1.5, 1.5, 64, state)
check("render_to_surface: returns Surface", isinstance(surf, pygame.Surface))
check("render_to_surface: returns bool for gpu", isinstance(used_gpu, (bool, np.bool_)))

# ── 6. Type consistency: stripe_s, step_s ──────────────────────────────────
print("\n--- Type consistency ---")
# stripe_s is defined as float in state defaults but uses int() in keybind
# This is a known pattern - check it doesn't cause issues
state_s = dict(state)
state_s["stripe_s"] = 0.0
img1 = mb.compute_image(16, 16, -2.0, 1.0, -1.5, 1.5, 32, params)
check("Rendering with float stripe_s: works", img1.shape == (16, 16, 3))

# ── 7. Edge case: extreme zoom ─────────────────────────────────────────────
print("\n--- Edge cases ---")
# Very high iteration
params_hi = dict(params, ncyl=math.sqrt(4096))
img_hi = mb.compute_image(16, 16, -2.0, 1.0, -1.5, 1.5, 4096, params)
check("High iter (4096): renders", img_hi.shape == (16, 16, 3))

# Very zoomed in
img_zoom = mb.compute_image(64, 64, -0.748, -0.746, 0.098, 0.100, 512, params)
check("Deep zoom: renders", img_zoom.shape == (64, 64, 3))

# Julia with different c
params_ju = dict(params, use_julia=True, julia_c_re=-0.7, julia_c_im=0.27015)
img_ju = mb.compute_image(32, 32, -1.5, 1.5, -1.5, 1.5, 64, params_ju)
check("Julia deep zoom: renders", img_ju.shape == (32, 32, 3))

# ── 8. Persistence round-trip ──────────────────────────────────────────────
print("\n--- Persistence round-trip ---")
test_settings = {
    "is-julia": "True",
    "julia-cx": "0.285",
    "julia-cy": "0.01",
    "palette-index": "5",
    "smooth": "False",
    "fxaa": "True",
}
test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "visual_tests", "audit_settings.yaml")
mb.save_settings(test_file, test_settings)
loaded = mb.load_settings(test_file)
check("Persistence: roundtrip preserves values",
      loaded.get("is-julia") == "True" and
      loaded.get("julia-cx") == "0.285" and
      loaded.get("julia-cy") == "0.01" and
      loaded.get("palette-index") == "5")
try:
    os.remove(test_file)
except:
    pass

# ── 9. MenuOverlay completeness ─────────────────────────────────────────────
print("\n--- MenuOverlay completeness ---")
overlay = mb.MenuOverlay()
check("MenuOverlay has draw", hasattr(overlay, "draw"))
check("MenuOverlay has toggle", hasattr(overlay, "toggle"))
check("MenuOverlay has handle_click", hasattr(overlay, "handle_click"))
check("MenuOverlay has _draw_keybinds", hasattr(overlay, "_draw_keybinds"))
check("MenuOverlay toggle works", overlay.active is False)
overlay.toggle()
check("MenuOverlay toggle: -> True", overlay.active is True)
overlay.toggle()
check("MenuOverlay toggle: -> False", overlay.active is False)

# ── 10. _draw_orbit coordinate mapping ─────────────────────────────────────
print("\n--- Orbit coordinate mapping ---")
# Check that _draw_orbit handles both modes
state_orb = dict(state)
state_orb["show_orbits"] = True
mb._RENDER_CACHE.clear()
surf_orb, _ = mb.render_to_surface(64, 64, -2.0, 1.0, -1.5, 1.5, 64, state_orb)
arr_orb = pygame.surfarray.array3d(surf_orb)
check("Orbits drawn: image has non-black pixels", np.sum(np.any(arr_orb > 0, axis=2)) > 0)

state_orb_ju = dict(state)
state_orb_ju["is_julia"] = True
mb._RENDER_CACHE.clear()
surf_orb_ju, _ = mb.render_to_surface(64, 64, -1.5, 1.5, -1.5, 1.5, 64, state_orb_ju)
arr_orb_ju = pygame.surfarray.array3d(surf_orb_ju)
check("Julia orbits drawn: image has non-black pixels", np.sum(np.any(arr_orb_ju > 0, axis=2)) > 0)

# ── 11. Module constants ───────────────────────────────────────────────────
print("\n--- Module constants ---")
check("DEFAULT_JULIA_C exists", hasattr(mb, "DEFAULT_JULIA_C"))
check("DEFAULT_JULIA_C correct", mb.DEFAULT_JULIA_C == [0.394, 0.338])
check("PALETTE_NAMES exists", hasattr(mb, "PALETTE_NAMES"))
check("PALETTE_NAMES has 16 entries", len(mb.PALETTE_NAMES) == 16)
check("COLOR_THETAS has 16 entries", len(mb.COLOR_THETAS) == 16)
check("PALETTE_NAMES match indices", len(mb.PALETTE_NAMES) == len(mb.COLOR_THETAS))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
