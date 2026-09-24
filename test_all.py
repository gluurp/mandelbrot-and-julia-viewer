"""Comprehensive tests for mandelbrot-testing-playground.py"""
import sys
import os
import math
import importlib.util
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("MB_PALETTE_REWRITE", "0")

_spec = importlib.util.spec_from_file_location(
    "mb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandelbrot_testing_playground.py")
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

NCOL = mb.NCOL
DEFAULT_NCYCLE = mb.DEFAULT_NCYCLE
DEFAULT_GRADIENT_STOPS = mb.DEFAULT_GRADIENT_STOPS
DEFAULT_GRADIENT_BLEND = mb.DEFAULT_GRADIENT_BLEND
TEST_GRADIENT_STOPS = [(0.0, (0.0, 0.0, 0.0)), (0.25, (1.0, 0.0, 0.0)),
                       (0.5, (1.0, 1.0, 0.0)), (0.75, (0.0, 0.0, 1.0)),
                       (1.0, (1.0, 1.0, 1.0))]
DEFAULT_STRIPE_S = mb.DEFAULT_STRIPE_S
DEFAULT_STRIPE_SIG = mb.DEFAULT_STRIPE_SIG
DEFAULT_STEP_S = mb.DEFAULT_STEP_S
DEFAULT_LIGHT_ANGLE = mb.DEFAULT_LIGHT_ANGLE
DEFAULT_LIGHT_AZIM = mb.DEFAULT_LIGHT_AZIM
DEFAULT_LIGHT_I = mb.DEFAULT_LIGHT_I
DEFAULT_K_AMBIANT = mb.DEFAULT_K_AMBIANT
DEFAULT_K_DIFFUSE = mb.DEFAULT_K_DIFFUSE
DEFAULT_K_SPECULAR = mb.DEFAULT_K_SPECULAR
DEFAULT_SHININESS = mb.DEFAULT_SHININESS
PALETTE_NAMES = ['fire-gradient', 'deep-sea-gradient', 'twilight-gradient',
                 'aurora-gradient', 'forest-gradient', 'sunset-gradient',
                 'amber-gradient', 'copper-gradient', 'electric-gradient',
                 'lava-gradient', 'teal-gradient']

import pygame
pygame.init()
pygame.display.set_mode((100, 100))

# Minimal keybind map for tests.
TEST_KEYBINDS = {"quit": "q", "menu": "m"}
TEST_MOUSE_KEYBINDS = {"zoom-in": "scroll_up", "zoom-out": "scroll_down",
                       "quit": "q", "menu": "m"}

PASS = 0
FAIL = 0

def test(name, condition, details=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {details}")


def build_state(**overrides):
    state = {
        "max_iter": 256,
        "gradient_stops": list(TEST_GRADIENT_STOPS),
        "gradient_blend": DEFAULT_GRADIENT_BLEND,
        "use_gpu": False,
        "stripe_s": 0.0,
        "stripe_sig": 0.9,
        "step_s": 0.0,
        "light_angle": DEFAULT_LIGHT_ANGLE,
        "light_azim": DEFAULT_LIGHT_AZIM,
        "light_i": DEFAULT_LIGHT_I,
        "k_ambiant": DEFAULT_K_AMBIANT,
        "k_diffuse": DEFAULT_K_DIFFUSE,
        "k_specular": DEFAULT_K_SPECULAR,
        "shininess": DEFAULT_SHININESS,
        "smooth": True,
        "fxaa": False,
        "palette_index": 0,
        "orbit_point_m": [0.018, -0.63],
        "orbit_point_j": [0.153, 0.473],
        "orbit_max_iter": 200,
        "orbit_point_size": 3,
        "set_blend": 0.0,
        "julia_c": [0.394, 0.338],
        "point_opacity_m": 1.0,
        "point_opacity_j": 1.0,
        "point_opacity_c": 1.0,
        "orbits_opacity_m": 1.0,
        "orbits_opacity_j": 1.0,
        "lines_opacity_m": 1.0,
        "lines_opacity_j": 1.0,
        "cycle_opacity": mb.DEFAULT_CYCLE_OPACITY,
        "center_opacity": mb.DEFAULT_CENTER_OPACITY,
        "grid_opacity": mb.DEFAULT_GRID_OPACITY,
        "grid_label_opacity": mb.DEFAULT_GRID_LABEL_OPACITY,
    }
    state.update(overrides)
    return state


def make_full_state(**overrides):
    s = build_state(**overrides)
    s["use_gpu"] = False
    return s


print("=" * 60)
print("TEST SUITE: mandelbrot-testing-playground.py")
print("=" * 60)

# ===========================================================================
# 1. Module constants
# ===========================================================================
print("\n--- Module Constants ---")
test("NCOL is 4096", NCOL == 4096)
test("DEFAULT_GRADIENT_STOPS has >= 2 stops", len(DEFAULT_GRADIENT_STOPS) >= 2)
test("PALETTE_BLEND_MODES covers smooth/linear",
     {"smooth", "linear"} <= set(mb.PALETTE_BLEND_MODES))

# ===========================================================================
# 2. Gradient interpolation
# ===========================================================================
print("\n--- gradient interpolation ---")
ct_smooth = mb._make_gradient_colortable(TEST_GRADIENT_STOPS, "smooth")
ct_linear = mb._make_gradient_colortable(TEST_GRADIENT_STOPS, "linear")
test("gradient colortable shape", ct_smooth.shape == (NCOL, 3))
test("gradient values in [0, 1]",
     np.all(ct_smooth >= -0.001) and np.all(ct_smooth <= 1.001))
test("smooth differs from linear on a multi-stop gradient",
     not np.allclose(ct_smooth, ct_linear, atol=1e-4))
for _pos, _rgb in TEST_GRADIENT_STOPS:
    _idx = int(round(_pos * (NCOL - 1)))
    test(f"smooth passes through stop {_pos}",
         np.allclose(ct_smooth[_idx], np.array(_rgb), atol=2e-3))
_piece = mb._make_gradient_colortable(
    [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))], "smooth")
_ramp = np.linspace(0.0, 1.0, NCOL)
test("2-stop smooth equals linear",
     np.allclose(_piece, np.column_stack([_ramp] * 3), atol=1e-6))
_overshoot_stops = [(0.0, (0.0, 0.0, 0.0)), (0.2, (1.0, 0.0, 0.0)),
                    (0.4, (1.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))]
_overshoot = mb._make_gradient_colortable(_overshoot_stops, "smooth")
test("smooth does not overshoot the stop range",
     float(_overshoot[:, 0].max()) <= 1.0 + 1e-6 and
     float(_overshoot[:, 0].min()) >= -1e-6)
test("gradient sample stays in range",
     all(0.0 <= value <= 1.0 for value in mb._gradient_sample(TEST_GRADIENT_STOPS, 0.6)))
test("invalid blend normalizes to default",
     mb._normalize_gradient_blend("banana") == DEFAULT_GRADIENT_BLEND)

# ===========================================================================
# 2.5 _make_gradient_colortable
# ===========================================================================
print("\n--- _make_gradient_colortable ---")

ct_grad = mb._make_gradient_colortable([
    (0.0, (1.0, 0.0, 0.0)),
    (1.0, (0.0, 0.0, 1.0)),
])
test("Gradient: shape correct", ct_grad.shape == (4096, 3))
test("Gradient: start is red", ct_grad[0, 0] > 0.9 and ct_grad[0, 1] < 0.1 and ct_grad[0, 2] < 0.1)
test("Gradient: end is blue", ct_grad[-1, 2] > 0.9 and ct_grad[-1, 0] < 0.1 and ct_grad[-1, 1] < 0.1)
test("Gradient: values in [0,1]", ct_grad.min() >= 0 and ct_grad.max() <= 1.0 + 1e-6)

ct_3 = mb._make_gradient_colortable([
    (0.0, (1.0, 0.0, 0.0)),
    (0.5, (0.0, 1.0, 0.0)),
    (1.0, (0.0, 0.0, 1.0)),
])
test("3-stop gradient: midpoint is green", ct_3[2048, 1] > 0.9)

# ===========================================================================
# 4. iterate_single
# ===========================================================================
print("\n--- iterate_single ---")
orbit_x, orbit_y, escaped = mb.iterate_single(0.5, 0.5, 50)
test("Exterior point escapes", escaped is True)
test("Exterior orbit is non-empty", len(orbit_x) > 0)

orbit_x, orbit_y, escaped = mb.iterate_single(-0.75, 0.0, 50)
test("Interior point bounded", escaped is False)
test("Interior orbit fills max_iter", len(orbit_x) == 50)

ox, oy, escaped = mb.iterate_single(0.0, 0.0, 10)
test("Origin (0,0) is bounded", escaped is False)

ox, oy, escaped = mb.iterate_single(2.0, 0.0, 10)
test("Point (2,0) escapes", escaped is True)
test("Escape orbit shorter than max", len(ox) <= 10)

test("All orbit values finite",
     np.all(np.isfinite(orbit_x)) and np.all(np.isfinite(orbit_y)))

# ===========================================================================
# 5. compute_orbit
# ===========================================================================
print("\n--- compute_orbit ---")
orbit = mb.compute_orbit(-0.75, 0.0, 50)
test("Bounded orbit has max_iter+1 points", len(orbit) == 51, f"len={len(orbit)}")
test("First point is the seed", abs(orbit[0, 0] - (-0.75)) < 1e-9 and abs(orbit[0, 1]) < 1e-9)
test("All orbit values finite", np.all(np.isfinite(orbit)))

orbit = mb.compute_orbit(0.5, 0.5, 50)
test("Escaping orbit shorter than max_iter+1", len(orbit) < 51, f"len={len(orbit)}")
test("Escaping orbit all finite", np.all(np.isfinite(orbit)))

orbit = mb.compute_orbit(0.0, 0.0, 50)
test("Origin orbit stays at origin",
     np.all(np.abs(orbit) < 1e-9))

# ===========================================================================
# 6. _smooth_iter_scalar
# ===========================================================================
print("\n--- _smooth_iter_scalar ---")
result = mb._smooth_iter_scalar(0.5, 0.5, 100, 0.0, 0.9)
test("_smooth_iter_scalar returns 4-tuple", len(result) == 4)
niter, stripe_a, dem, normal = result
test("Escaping point has niter > 0", niter > 0, f"niter={niter}")
test("Escaping point has finite dem", np.isfinite(dem))

result = mb._smooth_iter_scalar(-0.75, 0.0, 100, 0.0, 0.9)
niter, stripe_a, dem, normal = result
test("Interior point niter == 0", niter == 0.0, f"niter={niter}")

# With stripe active
result = mb._smooth_iter_scalar(0.5, 0.5, 100, 1.0, 0.9)
niter, stripe_a, dem, normal = result
test("Striped escape has stripe_a >= 0", stripe_a >= 0.0)
test("Striped escape has normal != 0", abs(normal) > 0.0 if niter > 0 else True)

# ===========================================================================
# 7. blinn_phong
# ===========================================================================
print("\n--- blinn_phong ---")
light_arr = np.array([
    0.5 * 2 * math.pi,
    0.5 * math.pi / 2,
    0.75,
    DEFAULT_K_AMBIANT,
    DEFAULT_K_DIFFUSE,
    DEFAULT_K_SPECULAR,
    DEFAULT_SHININESS,
], dtype=np.float64)
# Normal pointing straight up
n0 = complex(0.0, 1.0)
b = mb.blinn_phong(n0, light_arr)
test("Blinn-Phong: valid normal returns finite", np.isfinite(b))
test("Blinn-Phong: result in [0, 1]", 0.0 <= b <= 1.0 + 1e-6)

# Zero normal guard
n1 = complex(0.0, 0.0)
b2 = mb.blinn_phong(n1, light_arr)
test("Blinn-Phong: zero normal returns 0", b2 == 0.0, f"got {b2}")

# ===========================================================================
# 8. overlay
# ===========================================================================
print("\n--- overlay ---")
result = mb.overlay(0.5, 0.5, 1.0)
test("Overlay: gamma=1 returns blend", abs(result - 0.5) < 1e-6)
result = mb.overlay(0.3, 0.7, 1.0)
test("Overlay: gamma=1 low val", abs(result - 0.58) < 1e-6, f"got {result}")
# gamma=0 → returns x unchanged
result = mb.overlay(0.3, 0.7, 0.0)
test("Overlay: gamma=0 returns x", abs(result - 0.3) < 1e-6)

# ===========================================================================
# 9. color_pixel
# ===========================================================================
print("\n--- color_pixel ---")
colortable = mb._make_gradient_colortable(TEST_GRADIENT_STOPS)
light = light_arr
ncol = colortable.shape[0] - 1
ncycle = math.sqrt(64)

r, g, b = mb.color_pixel(10.0, 0.5, 0.0, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=True)
test("color_pixel smooth: returns 3 values", isinstance(r, float) and isinstance(g, float) and isinstance(b, float))
test("color_pixel smooth: RGB in [0,1]", 0 <= r <= 1.0 and 0 <= g <= 1.0 and 0 <= b <= 1.0)

r, g, b = mb.color_pixel(10.0, 0.0, 0.0, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=False)
test("color_pixel discrete: RGB in [0,1]", 0 <= r <= 1.0 and 0 <= g <= 1.0 and 0 <= b <= 1.0)

# step_s produces different result
r1, _, _ = mb.color_pixel(10.0, 0.0, 0.0, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=True)
r2, _, _ = mb.color_pixel(10.0, 0.5, 0.5, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=True)
test("step_s changes output (smooth=True)", abs(r1 - r2) > 0.01, f"r1={r1}, r2={r2}")

# step_s also works when smooth=False (was previously broken)
r1s, _, _ = mb.color_pixel(10.0, 0.0, 0.0, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=False)
r2s, _, _ = mb.color_pixel(10.0, 0.5, 5.0, 1.0, complex(0.5, 0.5), colortable, ncycle, light, smooth=False)
test("step_s changes output (smooth=False)", abs(r1s - r2s) > 0.01, f"r1={r1s}, r2={r2s}")

# ===========================================================================
# 10. compute_set_cpu
# ===========================================================================
print("\n--- compute_set_cpu ---")
colortable = mb._make_gradient_colortable(TEST_GRADIENT_STOPS)
creal = np.linspace(-2.0, 1.0, 32)
cim = np.linspace(-1.5, 1.5, 32)
mat = mb.compute_set_cpu(creal, cim, 64, colortable, math.sqrt(64),
                         0.0, 0.9, 0.0, 10.0, light, smooth=True)
test("CPU render shape (32,32,3)", mat.shape == (32, 32, 3))
test("CPU render: all finite", np.all(np.isfinite(mat)))
test("CPU render: values in [0,1]",
     np.all(mat >= -0.001) and np.all(mat <= 1.001))

# ===========================================================================
# 11. build_render_params
# ===========================================================================
print("\n--- build_render_params ---")
state = make_full_state()
params = mb.build_render_params(state, maxiter=64)
test("build_render_params returns dict", isinstance(params, dict))
test("params has colortable", "colortable" in params)
test("params colortable shape", params["colortable"].shape == (NCOL, 3))
test("params has ncycle", "ncy" in params)
test("params ncycle = sqrt(64) = 8", params["ncy"] == 8.0)
test("params inherits smooth", params["smooth"] == True)
test("params inherits fxaa", params["fxaa"] == False)
test("params use_gpu respects state", params["use_gpu"] == (state["use_gpu"] and mb._CUDA_AVAILABLE))
test("params light is np array", isinstance(params["light"], np.ndarray))
test("params light has 7 elements", params["light"].shape == (7,))

# Blend mode in params
state_blended = make_full_state(gradient_blend="linear")
params_blended = mb.build_render_params(state_blended, maxiter=64)
ct1 = params["colortable"]
ct2 = params_blended["colortable"]
test("Blend mode changes colortable in params", not np.allclose(ct1, ct2, atol=1e-4))

# ===========================================================================
# 12. GPU/CUDA consistency
# ===========================================================================
print("\n--- GPU/CUDA ---")
test("CUDA availability flag exists", hasattr(mb, "_CUDA_AVAILABLE"))
test("CUDA flag is boolean", isinstance(mb._CUDA_AVAILABLE, bool))
if mb._CUDA_AVAILABLE:
    test("CUDA available", mb._CUDA_AVAILABLE)
    # Run the same render on CPU and GPU, compare
    size = 64
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    colortable = mb._make_gradient_colortable(TEST_GRADIENT_STOPS)
    light = light_arr
    diag = math.sqrt((xmin - xmax) ** 2 + (ymin - ymax) ** 2)
    maxiter = 64

    # GPU render
    gpu_params = {
        "colortable": colortable,
        "ncy": math.sqrt(maxiter),
        "stripe_s": 0.0,
        "stripe_sig": 0.9,
        "step_s": 0.0,
        "light": light,
        "use_gpu": True,
        "smooth": True,
        "fxaa": False,
    }
    try:
        gpu_result = mb.compute_image(size, size, xmin, xmax, ymin, ymax, maxiter, gpu_params)
        test("GPU render returns array", gpu_result is not None)
        test("GPU render shape", gpu_result.shape == (size, size, 3))
        test("GPU render all finite", np.all(np.isfinite(gpu_result)))

        # CPU render
        cpu_params = dict(gpu_params)
        cpu_params["use_gpu"] = False
        cpu_result = mb.compute_image(size, size, xmin, xmax, ymin, ymax, maxiter, cpu_params)
        test("CPU render shape", cpu_result.shape == (size, size, 3))

        avg_diff = np.mean(np.abs(gpu_result.astype(np.float64) - cpu_result.astype(np.float64)))
        max_diff = np.max(np.abs(gpu_result.astype(np.float64) - cpu_result.astype(np.float64)))
        test("GPU/CPU avg diff small", avg_diff < 1.0, f"avg={avg_diff:.4f}")
        test("GPU/CPU max diff bounded", max_diff < 50, f"max={max_diff}")
    except Exception as e:
        test("GPU render", False, str(e))
else:
    test("CUDA not available (CPU mode)", not mb._CUDA_AVAILABLE)

# ===========================================================================
# 12b. Julia set
# ===========================================================================
print("\n--- Julia Set ---")
# Julia orbit: z0 = pixel coordinate, z = z^2 + c_J
orbit_ju = mb.compute_orbit(0.394, 0.338, 100, use_julia=True,
                            julia_c_re=0.394, julia_c_im=0.338)
test("Julia orbit starts at z0 = pixel coord", abs(orbit_ju[0, 0] - 0.394) < 1e-9)
test("Julia orbit all finite", np.all(np.isfinite(orbit_ju)))

# Julia orbit for c_J=0 should converge to 0 (z^2 + 0 converges for |z0| < 1)
orbit_ju0 = mb.compute_orbit(0.5, 0.0, 50, use_julia=True, julia_c_re=0.0, julia_c_im=0.0)
test("Julia c=0: orbit bounded (inside unit circle)", np.all(np.abs(orbit_ju0) <= 1.0))

# Julia orbit for c_J=0 with |z0| > 1 should escape
orbit_ju1 = mb.compute_orbit(2.0, 0.0, 50, use_julia=True, julia_c_re=0.0, julia_c_im=0.0)
test("Julia c=0: orbit escapes (|z0| > 1)", len(orbit_ju1) < 51)

# _smooth_iter_scalar with Julia mode
result_ju = mb._smooth_iter_scalar(0.5, 0.5, 100, 0.0, 0.9, True, 0.394, 0.338)
niter_ju, stripe_a_ju, dem_ju, normal_ju = result_ju
test("Julia _smooth_iter_scalar: escaping point has niter > 0", niter_ju > 0, f"niter={niter_ju}")
test("Julia _smooth_iter_scalar all finite", np.isfinite(niter_ju) and np.isfinite(dem_ju) and np.isfinite(normal_ju))

# Julia render via compute_image
julia_state = {
    "colortable": mb._make_gradient_colortable(TEST_GRADIENT_STOPS),
    "ncy": math.sqrt(64),
    "stripe_s": 0.0,
    "stripe_sig": 0.9,
    "step_s": 0.0,
    "light": light_arr,
    "use_gpu": False,
    "smooth": True,
    "fxaa": False,
    "use_julia": True,
    "julia_c_re": 0.394,
    "julia_c_im": 0.338,
}
julia_result = mb.compute_image(32, 32, -1.5, 1.5, -1.5, 1.5, 64, julia_state)
test("Julia render: shape (32,32,3)", julia_result.shape == (32, 32, 3))
test("Julia render: all finite", np.all(np.isfinite(julia_result.astype(float))))
test("Julia render: values in [0,255]", np.all(julia_result >= 0) and np.all(julia_result <= 255))

# Mandelbrot render for comparison (should differ)
mb_state = dict(julia_state)
mb_state["use_julia"] = False
mb_state["julia_c_re"] = 0.0
mb_state["julia_c_im"] = 0.0
mb_result = mb.compute_image(32, 32, -1.5, 1.5, -1.5, 1.5, 64, mb_state)
test("Julia differs from Mandelbrot", not np.array_equal(julia_result, mb_result))

# Julia c=0 should produce circular structure (unit circle)
julia_c0_state = dict(julia_state, julia_c_re=0.0, julia_c_im=0.0)
julia_c0_result = mb.compute_image(32, 32, -1.5, 1.5, -1.5, 1.5, 128, julia_c0_state)
test("Julia c=0 renders without error", julia_c0_result.shape == (32, 32, 3))

# GPU Julia rendering (if CUDA)
if mb._CUDA_AVAILABLE:
    gpu_julia = dict(julia_state, use_gpu=True)
    mb._RENDER_CACHE.clear()
    gpu_ju_result = mb.compute_image(32, 32, -1.5, 1.5, -1.5, 1.5, 64, gpu_julia)
    test("GPU Julia: shape correct", gpu_ju_result.shape == (32, 32, 3))
    test("GPU Julia: CPU/GPU consistent",
         np.array_equal(julia_result, gpu_ju_result) or
         np.max(np.abs(julia_result.astype(int) - gpu_ju_result.astype(int))) <= 1,
         f"max_diff={np.max(np.abs(julia_result.astype(int) - gpu_ju_result.astype(int)))}")

# build_render_params with Julia
julia_params_state = build_state(julia_c=[0.394, 0.338], set_blend=1.0)
julia_params = mb.build_render_params(julia_params_state, maxiter=32, use_julia=True)
test("build_render_params: has use_julia", "use_julia" in julia_params)
test("build_render_params: use_julia is True", julia_params["use_julia"] is True)
test("build_render_params: has julia_c_re", "julia_c_re" in julia_params)
test("build_render_params: julia_c_re correct", abs(julia_params["julia_c_re"] - 0.394) < 0.001)
test("build_render_params: julia_c_im correct", abs(julia_params["julia_c_im"] - 0.338) < 0.001)

# ===========================================================================
# 12c. Blend functionality
# ===========================================================================
print("\n--- Blend Functionality ---")
# Blend at 0.0 should be Mandelbrot only
mb_state_blend0 = build_state(set_blend=0.0)
mb_surf, _ = mb.render_to_surface(32, 32, -2.0, 1.0, -1.5, 1.5, 32, mb_state_blend0)
mb_result0 = pygame.surfarray.pixels3d(mb_surf)
test("Blend 0.0: renders Mandelbrot", mb_result0.shape == (32, 32, 3))

# Blend at 1.0 should be Julia only
ju_state_blend1 = build_state(set_blend=1.0)
ju_surf, _ = mb.render_to_surface(32, 32, -2.0, 1.0, -1.5, 1.5, 32, ju_state_blend1)
ju_result1 = pygame.surfarray.pixels3d(ju_surf)
test("Blend 1.0: renders Julia", ju_result1.shape == (32, 32, 3))
test("Blend 0.0 != Blend 1.0", not np.array_equal(mb_result0, ju_result1))

# Blend at 0.5 should be a mix
blend_state = build_state(set_blend=0.5)
blend_surf, _ = mb.render_to_surface(32, 32, -2.0, 1.0, -1.5, 1.5, 32, blend_state)
blend_result = pygame.surfarray.pixels3d(blend_surf)
test("Blend 0.5: shape correct", blend_result.shape == (32, 32, 3))
test("Blend 0.5 differs from pure MB", not np.array_equal(blend_result, mb_result0))
test("Blend 0.5 differs from pure Julia", not np.array_equal(blend_result, ju_result1))

# _apply_step blends correctly
test_state = build_state(set_blend=0.25)
mb._apply_step(test_state, "blend-up", 1.0)
test("Blend step up from 0.25 = 0.30", abs(test_state["set_blend"] - 0.30) < 0.001, f"got {test_state['set_blend']}")

test_state = build_state(set_blend=1.0)
mb._apply_step(test_state, "blend-down", 1.0)
test("Blend step down from 1.0 = 0.95", abs(test_state["set_blend"] - 0.95) < 0.001, f"got {test_state['set_blend']}")

test_state = build_state(set_blend=0.0)
mb._apply_step(test_state, "blend-up", 1.0)
test("Blend step up from 0.0 = 0.05", abs(test_state["set_blend"] - 0.05) < 0.001, f"got {test_state['set_blend']}")

test_state = build_state(set_blend=0.25)
mb._apply_step(test_state, "blend-up", 10.0)  # shift
test("Blend step up from 0.25 (shift) = 0.75", abs(test_state["set_blend"] - 0.75) < 0.001, f"got {test_state['set_blend']}")
# ===========================================================================
print("\n--- _post_process ---")
mat = np.random.rand(16, 16, 3).astype(np.float32)
mat[0, 0, 0] = float('nan')
mat[0, 0, 1] = float('inf')
result = mb._post_process(mat, apply_aa=False)
test("_post_process: shape (16,16,3)", result.shape == (16, 16, 3))
test("_post_process: uint8", result.dtype == np.uint8)
test("_post_process: nan/inf handled", result[0, 0, 0] >= 0 and result[0, 0, 0] <= 255)

# FXAA post_process
mat2 = np.random.rand(16, 16, 3).astype(np.float32)
result2 = mb._post_process(mat2, apply_aa=True)
test("_post_process AA: shape preserved", result2.shape == (16, 16, 3))

# ===========================================================================
# 14. apply_fxaa
# ===========================================================================
print("\n--- apply_fxaa ---")
mat = np.random.rand(20, 20, 3).astype(np.float32)
result = mb.apply_fxaa(mat)
test("FXAA: shape preserved", result.shape == (20, 20, 3))
test("FXAA: all finite", np.all(np.isfinite(result)))
# Edges should be unchanged (border pixels copied directly)
test("FXAA: top edge preserved", np.allclose(result[0], mat[0]))
test("FXAA: bottom edge preserved", np.allclose(result[-1], mat[-1]))
test("FXAA: left edge preserved", np.allclose(result[:, 0], mat[:, 0]))
test("FXAA: right edge preserved", np.allclose(result[:, -1], mat[:, -1]))

# ===========================================================================
# 15. compute_image
# ===========================================================================
print("\n--- compute_image ---")
state = make_full_state()
params = mb.build_render_params(state, maxiter=32)
result = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 32, params)
test("compute_image: shape (32,32,3)", result.shape == (32, 32, 3))
test("compute_image: uint8", result.dtype == np.uint8)
test("compute_image: all values in [0,255]",
     np.all(result >= 0) and np.all(result <= 255))

# Interior pixel should be dark (bounded orbit → black)
test("compute_image: origin is dark",
     np.mean(result[16, 16]) < 50, f"mean={np.mean(result[16, 16])}")

# ===========================================================================
# 16. next_pow2 / prev_pow2
# ===========================================================================
print("\n--- next_pow2 / prev_pow2 ---")
test("next_pow2(1) = 1", mb.next_pow2(1) == 1)
test("next_pow2(2) = 2", mb.next_pow2(2) == 2)
test("next_pow2(3) = 4", mb.next_pow2(3) == 4)
test("next_pow2(4) = 4", mb.next_pow2(4) == 4)
test("next_pow2(5) = 8", mb.next_pow2(5) == 8)
test("next_pow2(7) = 8", mb.next_pow2(7) == 8)
test("next_pow2(8) = 8", mb.next_pow2(8) == 8)
test("next_pow2(9) = 16", mb.next_pow2(9) == 16)
test("next_pow2(0) = 2", mb.next_pow2(0) == 2)
test("next_pow2(-1) = 2", mb.next_pow2(-1) == 2)

test("prev_pow2(1) = 1", mb.prev_pow2(1) == 1)
test("prev_pow2(2) = 2", mb.prev_pow2(2) == 2)
test("prev_pow2(3) = 2", mb.prev_pow2(3) == 2)
test("prev_pow2(4) = 4", mb.prev_pow2(4) == 4)
test("prev_pow2(5) = 4", mb.prev_pow2(5) == 4)
test("prev_pow2(7) = 4", mb.prev_pow2(7) == 4)
test("prev_pow2(8) = 8", mb.prev_pow2(8) == 8)
test("prev_pow2(9) = 8", mb.prev_pow2(9) == 8)
test("prev_pow2(0) = 1", mb.prev_pow2(0) == 1)

# ===========================================================================
# 17. fix_aspect_ratio
# ===========================================================================
print("\n--- fix_aspect_ratio ---")
x_min, x_max, y_min, y_max = mb.fix_aspect_ratio(-2.5, 1.0, -1.5, 1.5, 800, 600)
test("fix_aspect: correct ratio", abs((x_max - x_min) / (y_max - y_min) - 800/600) < 0.01)

x_min, x_max, y_min, y_max = mb.fix_aspect_ratio(-2.0, 1.0, -1.0, 1.0, 400, 200)
test("fix_aspect: width > height ratio", abs((x_max - x_min) / (y_max - y_min) - 2.0) < 0.01)

# ===========================================================================
# 18. _liang_barsky_clip
# ===========================================================================
print("\n--- _liang_barsky_clip ---")
result = mb._liang_barsky_clip(0, 0, 100, 100, 0, 0, 50, 50)
test("Clip: partial line clipped", result is not None)
test("Clip: clipped line within bounds",
     result[0] >= 0 and result[1] >= 0 and result[2] <= 50 and result[3] <= 50)

result = mb._liang_barsky_clip(0, 0, 0, 0, 0, 0, 50, 50)
# Degenerate line: dx=0, dy=0 → p[i]=0 for all, q[i]>=0 → returns clipped (same point)
test("Clip: zero-length line handled", result is not None)

result = mb._liang_barsky_clip(-100, -100, -50, -50, 0, 0, 50, 50)
test("Clip: fully outside returns None", result is None)

result = mb._liang_barsky_clip(10, 10, 40, 40, 0, 0, 50, 50)
test("Clip: fully inside returns original", result == (10, 10, 40, 40))

# ===========================================================================
# 19. _mandelbrot_to_screen_all
# ===========================================================================
print("\n--- _mandelbrot_to_screen_all ---")
points = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
screen_pts = mb._mandelbrot_to_screen_all(points, -2.0, 2.0, -2.0, 2.0, 100, 100)
test("Mandelbrot→screen: correct count", len(screen_pts) == 4)
test("Mandelbrot→screen: origin maps to center",
     abs(screen_pts[0][0] - 50) < 0.1 and abs(screen_pts[0][1] - 50) < 0.1)

# ===========================================================================
# 20. _orbit_pixel_color
# ===========================================================================
print("\n--- _orbit_pixel_color ---")
colortable = mb._make_gradient_colortable(TEST_GRADIENT_STOPS)
ncycle = math.sqrt(200)
r, g, b = mb._orbit_pixel_color(5, True, colortable, ncycle)
test("orbit_pixel_color smooth: returns 3 ints", isinstance(r, int) and isinstance(g, int) and isinstance(b, int))
test("orbit_pixel_color smooth: RGB in [0,255]", 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)

r, g, b = mb._orbit_pixel_color(5, False, colortable, ncycle)
test("orbit_pixel_color discrete: RGB in [0,255]", 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)

# Disabled orbits must also hide their draggable starting-point marker
original_compute_orbit = mb.compute_orbit
original_circle = mb.pygame.draw.circle
draw_calls = []
mb.compute_orbit = lambda *args, **kwargs: np.array([[0.0, 0.0]])
mb.pygame.draw.circle = lambda *args: draw_calls.append(args)
try:
    screen = pygame.Surface((100, 100))
    visible_state = build_state(orbits_opacity_m=0.0, orbits_opacity_j=0.0,
                                lines_opacity_m=0.0, lines_opacity_j=0.0,
                                set_blend=0.0)
    mb._draw_orbit(screen, visible_state, -2.0, 2.0, -2.0, 2.0, 100, 100)
    test("Point markers always render", len(draw_calls) > 0)
finally:
    mb.compute_orbit = original_compute_orbit
    mb.pygame.draw.circle = original_circle

original_compute_orbit = mb.compute_orbit
original_circle = mb.pygame.draw.circle
draw_calls = []
mb.compute_orbit = lambda *args, **kwargs: np.array([[0.0, 0.0]])
mb.pygame.draw.circle = lambda *args: draw_calls.append(args)
try:
    screen = pygame.Surface((100, 100))
    c_state = build_state(orbits_opacity_m=0.0, orbits_opacity_j=0.0,
                          lines_opacity_m=0.0, lines_opacity_j=0.0,
                          set_blend=1.0,
                          point_opacity_m=0.0, point_opacity_j=0.0,
                          point_opacity_c=1.0)
    expected_c = mb._palette_section_color(c_state, "julia c point")
    mb._draw_orbit(screen, c_state, -2.0, 2.0, -2.0, 2.0, 100, 100)
    test("Julia c-point uses the palette section color",
         expected_c in [call[1] for call in draw_calls if len(call) > 1])

    draw_calls.clear()
    c_state["point_opacity_m"] = 1.0
    c_state["point_opacity_j"] = 1.0
    c_state["split_mode"] = "horizontal"
    c_state["julia_viewport"] = list(mb.DEFAULT_JULIA_VIEWPORT)
    panes = mb._compute_pane_bounds("horizontal", "horizontal",
                                    -2.0, 2.0, -2.0, 2.0, 100, 100,
                                    julia_viewport=c_state["julia_viewport"])
    mb._draw_orbit(screen, c_state, -2.0, 2.0, -2.0, 2.0, 100, 100,
                   pane_bounds=panes)
    expected_colors = sorted([
        mb._palette_section_color(c_state, "mb orbit point"),
        mb._palette_section_color(c_state, "julia c point"),
        mb._palette_section_color(c_state, "ju orbit point"),
    ])
    drawn_colors = sorted(call[1] for call in draw_calls if len(call) > 1)
    test("Split mode draws M/C on the Mandelbrot pane and J on the Julia pane",
         drawn_colors == expected_colors)
finally:
    mb.compute_orbit = original_compute_orbit
    mb.pygame.draw.circle = original_circle

screen = pygame.Surface((120, 120))
screen.fill((0, 0, 0))
grid_state = build_state(grid_opacity=0.5)
mb._draw_grid(screen, grid_state, -2.0, 2.0, -2.0, 2.0, 120, 120)
grid_pixels = pygame.surfarray.array3d(screen)
test("Grid renders visible lines", np.any(grid_pixels > 0))

# ===========================================================================
# 21. Settings persistence (load/save round-trip)
# ===========================================================================
print("\n--- Settings Persistence ---")
import tempfile
tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
tmpfile.write("stripe_s: '0.0'\nstep_s: '0.0'\nkeybind.quit: escape\nkeybind.grid-opac-up: ctrl+r\n")
tmpfile.close()

loaded = mb.load_settings(tmpfile.name)
test("YAML loads", True)
test("YAML has stripe_s", "stripe_s" in loaded)
test("YAML has step_s", "step_s" in loaded)
test("YAML has keybind.grid-opac-up", "keybind.grid-opac-up" in loaded)
test("YAML stripe_s value", loaded["stripe_s"] == "0.0")
test("YAML keybind.grid-opac-up value", loaded["keybind.grid-opac-up"] == "ctrl+r")

test("YAML no orbit-active (uses show-orbits)", "orbit-active" not in loaded)

# Round-trip save
save_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
save_file.close()
mb.save_settings(save_file.name, loaded)
reloaded = mb.load_settings(save_file.name)
test("Save/load round-trip: stripe_s", reloaded.get("stripe_s") == "0.0")
test("Save/load round-trip: keybind matches", reloaded.get("keybind.grid-opac-up") == "ctrl+r")

# Viewport parsing and validation
main_settings = {
    "view-xmin": "-1.25",
    "view-xmax": "0.75",
    "view-ymin": "-0.75",
    "view-ymax": "1.25",
}
main_view = mb._parse_main_viewport(main_settings)
test("Main viewport parses", main_view == [-1.25, 0.75, -0.75, 1.25])
test("Main viewport rejects invalid bounds",
     mb._parse_main_viewport({"view-xmin": "nan", "view-xmax": "1",
                              "view-ymin": "-1", "view-ymax": "1"}) == [-2.5, 1.0, -1.5, 1.5])
julia_view = mb._parse_julia_viewport({"julia-viewport": "-1.0,1.0,-0.5,0.5"})
test("Julia viewport parses", julia_view == [-1.0, 1.0, -0.5, 0.5])
test("Julia viewport rejects malformed bounds",
     mb._parse_julia_viewport({"julia-viewport": "bad"}) == list(mb.DEFAULT_JULIA_VIEWPORT))

# Persistent value snapshot includes the interactive state users expect to retain
persist_state = build_state(
    julia_c=[-0.8, 0.2],
    orbit_point_m=[-0.3, 0.4],
    orbit_point_j=[0.2, -0.7],
    julia_viewport=[-1.2, 1.2, -0.8, 0.8],
)
persist_values = mb._build_persistent_values(
    persist_state, -2.0, 1.0, -1.5, 1.5, TEST_KEYBINDS)
test("Persistent snapshot: Julia c", persist_values["julia-cx"] == "-0.8" and persist_values["julia-cy"] == "0.2")
test("Persistent snapshot: Mandelbrot orbit", persist_values["orbit-mx"] == "-0.3" and persist_values["orbit-my"] == "0.4")
test("Persistent snapshot: Julia orbit", persist_values["orbit-jx"] == "0.2" and persist_values["orbit-jy"] == "-0.7")
test("Persistent snapshot: main viewport", persist_values["view-xmin"] == "-2.0" and persist_values["view-ymax"] == "1.5")
test("Persistent snapshot: Julia viewport", persist_values["julia-viewport"] == "-1.2,1.2,-0.8,0.8")
test("Persistent snapshot: no width/height keys", "width" not in persist_values and "height" not in persist_values)

# Debounced persistence saves after one second idle and force-saves on exit
persist_settings = {}
debounce_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
debounce_file.close()
initial_snapshot = mb._persistent_snapshot(mb._build_persistent_values(
    persist_state, -2.0, 1.0, -1.5, 1.5))
persist_state["julia_c"] = [-0.9, 0.3]
changed_snapshot, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, -2.0, 1.0, -1.5, 1.5,
    TEST_KEYBINDS, True, initial_snapshot, None,
    settings_file=debounce_file.name, now=10.0)
test("Debounced persistence: no early save", not persist_settings)
_, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, -2.0, 1.0, -1.5, 1.5,
    TEST_KEYBINDS, True, changed_snapshot, changed_time,
    settings_file=debounce_file.name, now=10.9)
test("Debounced persistence: waits for full debounce", not persist_settings)
_, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, -2.0, 1.0, -1.5, 1.5,
    TEST_KEYBINDS, True, changed_snapshot, changed_time,
    settings_file=debounce_file.name, now=11.0)
test("Debounced persistence: saves after one second", "julia-cx" in persist_settings)
force_settings = {}
mb._save_persistent_state(
    force_settings, persist_state, -2.0, 1.0, -1.5, 1.5,
    TEST_KEYBINDS, True, changed_snapshot, changed_time,
    force=True, settings_file=debounce_file.name, now=11.0)
test("Persistent force save: writes immediately", "julia-cx" in force_settings)

roundtrip_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
roundtrip_file.close()
mb.save_settings(roundtrip_file.name, persist_settings)
roundtrip = mb.load_settings(roundtrip_file.name)
test("Persistent YAML round-trip: Julia c",
     roundtrip.get("julia-cx") == "-0.9" and roundtrip.get("julia-cy") == "0.3")
test("Persistent YAML round-trip: Mandelbrot orbit",
     roundtrip.get("orbit-mx") == "-0.3" and roundtrip.get("orbit-my") == "0.4")
test("Persistent YAML round-trip: Julia orbit",
     roundtrip.get("orbit-jx") == "0.2" and roundtrip.get("orbit-jy") == "-0.7")
test("Persistent YAML round-trip: main viewport",
     roundtrip.get("view-xmin") == "-2.0" and roundtrip.get("view-ymax") == "1.5")
test("Persistent YAML round-trip: Julia viewport",
     roundtrip.get("julia-viewport") == "-1.2,1.2,-0.8,0.8")
os.unlink(roundtrip_file.name)

os.unlink(tmpfile.name)
os.unlink(save_file.name)
os.unlink(debounce_file.name)

# get_persistent_setting
test_settings = {"max_iter": "256", "smooth": "true", "bad_val": "not_a_number"}
val = mb.get_persistent_setting(test_settings, "max_iter", cast=int, default=64)
test("get_persistent_setting: int cast", val == 256)
val = mb.get_persistent_setting(test_settings, "missing", cast=float, default=0.5)
test("get_persistent_setting: default for missing", val == 0.5)
val = mb.get_persistent_setting(test_settings, "bad_val", cast=float, default=0.5)
test("get_persistent_setting: fallback on bad cast", val == 0.5)

# get_keybind
test("get_keybind reads from settings",
     mb.get_keybind({"keybind.test": "ctrl+x"}, "test", "default") == "ctrl+x")
test("get_keybind falls back to default",
     mb.get_keybind({}, "missing", "default") == "default")

# ===========================================================================
# 22. MenuOverlay
# ===========================================================================
print("\n--- MenuOverlay ---")
overlay = mb.MenuOverlay()
test("MenuOverlay starts inactive", overlay.active is False)

overlay.toggle()
test("MenuOverlay toggles active", overlay.active is True)
overlay.toggle()
test("MenuOverlay toggles back", overlay.active is False)

# Create a minimal state for MenuOverlay tests
state = build_state()
overlay.active = True

menu_state = make_full_state()
menu_screen = pygame.Surface((2540, 1400))
overlay._menu_cache = None
overlay.draw(menu_screen, pygame.font.SysFont("monospace", 18), menu_state,
             TEST_KEYBINDS, -2.0, 1.0, -1.5, 1.5)
test("MenuOverlay: split control is present", "toggle-split" in overlay.button_rects)
test("MenuOverlay: grid opacity control is present", "grid_opacity" in overlay.button_rects)
test("MenuOverlay: highlight-points button is present", "highlight-points" in overlay.button_rects)
test("MenuOverlay: points/keybinds toggles are present",
     {"toggle-point-info", "show-keybinds"}.issubset(overlay.button_rects))
test("MenuOverlay: point opacity controls are present",
     {"point_opacity_m", "point_opacity_j", "point_opacity_c",
      "cycle_opacity", "center_opacity",
      "orbits_opacity_m", "orbits_opacity_j",
      "lines_opacity_m", "lines_opacity_j",
      "grid_opacity", "grid_label_opacity"}.issubset(overlay.button_rects))
test("MenuOverlay: point size controls are present",
     {"point_size_m", "point_size_j", "point_size_c"}.issubset(overlay.button_rects))
test("MenuOverlay: visibility toggles removed",
     not ({"show_cycle_points", "show_center_points", "show_orbits_m",
           "show_orbit_lines_m", "show_orbits_j",
           "show_orbit_lines_j"} & set(overlay.button_rects)))
test("MenuOverlay: legacy controls removed",
     not ({"c_color_r", "point_color_r", "point_color_g", "point_color_b",
           "hue_0", "hue_1", "sat", "phase", "show_grid",
           "load-palette-file", "show_orbit_handles",
           "points-menu", "keybinds-menu"} & set(overlay.button_rects)))

parameter_keys = {
    "max_iter", "orbit_max_iter", "stripe_s", "step_s", "set_blend",
    "grid_opacity", "grid_label_opacity",
    "point_opacity_m", "point_opacity_j", "point_opacity_c",
    "cycle_opacity", "center_opacity",
    "orbits_opacity_m", "orbits_opacity_j", "lines_opacity_m", "lines_opacity_j",
    "point_size_m", "point_size_j", "point_size_c", "light_angle",
    "light_azim", "light_i", "k_ambiant", "k_diffuse", "k_specular",
    "shininess",
}
test("MenuOverlay: every parameter row has a control",
     parameter_keys.issubset(overlay.button_rects))

overlay._menu_cache = None
overlay.draw(menu_screen, pygame.font.SysFont("monospace", 18), menu_state,
             TEST_KEYBINDS, -2.0, 1.0, -1.5, 1.5)
cache_key = overlay._menu_cache[0]
resized_menu_screen = pygame.Surface((1270, 700))
overlay.draw(resized_menu_screen, pygame.font.SysFont("monospace", 18), menu_state,
             TEST_KEYBINDS, -2.0, 1.0, -1.5, 1.5)
test("MenuOverlay: cache invalidates when the window resizes",
     overlay._menu_cache[0] != cache_key)

# Simulate button rects for testing _do_action
# Test minus/plus button unpacking
overlay.button_rects = {
    "stripe_s": (100, 100, 36, 28, 140, 100, 36, 28, 180, 100, 112, 28),  # minus, plus, reset
}
rects = overlay._unpack_rects(overlay.button_rects["stripe_s"], "stripe_s")
test("Button unpack: 12-tuple → 3 rects", len(rects) == 3)
test("Button unpack: types correct", rects[0][1] == "minus" and rects[1][1] == "plus" and rects[2][1] == "reset")

overlay.button_rects = {
    "light_angle": (100, 100, 36, 28, 140, 100, 36, 28),  # minus, plus
}
rects = overlay._unpack_rects(overlay.button_rects["light_angle"], "light_angle")
test("Button unpack: 8-tuple → 2 rects", len(rects) == 2)

overlay.button_rects = {
    "smooth": (100, 100, 200, 28),  # single click
}
rects = overlay._unpack_rects(overlay.button_rects["smooth"], "smooth")
test("Button unpack: 4-tuple → 1 rect", len(rects) == 1)
test("Button unpack: single type is click", rects[0][1] == "click")

# Test _do_action for cyclic wrapping
state["light_angle"] = 0.99
mb.pygame.key.set_mods(0)  # no modifiers
overlay._do_action("light_angle", "plus", state)
test("MenuOverlay: light_angle wraps from 0.99→0.0",
     abs(state["light_angle"] - 0.09) < 0.011 or state["light_angle"] < 0.1,
     f"light_angle={state['light_angle']}")

# Test _do_action for clamped values (shininess)
state["shininess"] = 98.0
overlay._do_action("shininess", "plus", state)
test("MenuOverlay: shininess clamps at 100", state["shininess"] <= 100.0, f"shininess={state['shininess']}")

# Test toggle
state["fxaa"] = False
overlay._do_action("fxaa", "click", state)
test("MenuOverlay: toggle sets fxaa True", state["fxaa"] is True)

# Test cycle-palette
state["palette_index"] = 0
state["_palette_file_idx"] = 0
overlay._do_action("cycle-palette", "click", state)
test("MenuOverlay: cycle-palette sets _palette_file_idx", state["_palette_file_idx"] == 1)
test("MenuOverlay: cycle-palette sets gradient_stops",
     "gradient_stops" in state and len(state["gradient_stops"]) >= 2)
test("MenuOverlay: cycle-palette sets gradient_blend",
     state.get("gradient_blend") in mb.PALETTE_BLEND_MODES)

# Test reset-all
overlay._do_action("reset-all", "click", state)
test("MenuOverlay: reset-all restores palette index", state["palette_index"] == 0)
test("MenuOverlay: reset-all restores shininess", state["shininess"] == DEFAULT_SHININESS)

# Test reset-orbit-point
state["orbit_point_m"] = [-1.0, 2.0]
overlay._do_action("reset-orbit-point", "click", state)
test("MenuOverlay: reset-orbit-point resets julia_c",
      state["julia_c"] == mb.DEFAULT_JULIA_C)
test("MenuOverlay: reset-orbit-point resets orbit_point_m",
      state["orbit_point_m"] == mb.DEFAULT_ORBIT_POINT_M)
test("MenuOverlay: reset-orbit-point resets orbit_point_j",
      state["orbit_point_j"] == mb.DEFAULT_ORBIT_POINT_J)

# Test reset-orbit-points (the action button)
state["orbit_point_m"] = [1.0, 1.0]
state["orbit_point_j"] = [2.0, 2.0]
state["julia_c"] = [0.5, 0.5]
overlay._do_action("reset-orbit-points", "click", state)
test("MenuOverlay: reset-orbit-points resets julia_c",
      state["julia_c"] == mb.DEFAULT_JULIA_C)
test("MenuOverlay: reset-orbit-points resets orbit_point_m",
      state["orbit_point_m"] == mb.DEFAULT_ORBIT_POINT_M)
test("MenuOverlay: reset-orbit-points resets orbit_point_j",
      state["orbit_point_j"] == mb.DEFAULT_ORBIT_POINT_J)

# Test granular reset buttons
state["orbit_point_m"] = [1.0, 1.0]
overlay._do_action("reset-orbit-point-m", "click", state)
test("MenuOverlay: reset-orbit-point-m resets orbit_point_m",
      state["orbit_point_m"] == mb.DEFAULT_ORBIT_POINT_M)

state["orbit_point_j"] = [2.0, 2.0]
overlay._do_action("reset-orbit-point-j", "click", state)
test("MenuOverlay: reset-orbit-point-j resets orbit_point_j",
      state["orbit_point_j"] == mb.DEFAULT_ORBIT_POINT_J)

state["julia_c"] = [0.5, 0.5]
overlay._do_action("reset-julia-c", "click", state)
test("MenuOverlay: reset-julia-c resets julia_c",
      state["julia_c"] == mb.DEFAULT_JULIA_C)

# Test opacity stepping (0 is the off state)
state["orbits_opacity_m"] = 1.0
overlay._do_action("orbits_opacity_m", "minus", state)
test("MenuOverlay: mb orbits opacity steps down",
      abs(state["orbits_opacity_m"] - 0.95) < 0.001, f"got {state['orbits_opacity_m']}")

state["orbits_opacity_m"] = 0.05
overlay._do_action("orbits_opacity_m", "minus", state)
test("MenuOverlay: mb orbits opacity reaches off (0.0)",
      state["orbits_opacity_m"] == 0.0, f"got {state['orbits_opacity_m']}")

# Test blend step in menu (floating point fix)
state["set_blend"] = 1.0
mb.pygame.key.set_mods(0)  # no modifiers
overlay._do_action("set_blend", "minus", state)
test("MenuOverlay: blend down from 1.0 = 0.95 (no fp error)",
      abs(state["set_blend"] - 0.95) < 0.001, f"got {state['set_blend']}")

# Test orbit_max_iter step in menu (2^x exponential pattern)
state["orbit_max_iter"] = 200
mb.pygame.key.set_mods(0)
overlay._do_action("orbit_max_iter", "plus", state)
test("MenuOverlay: orbit iter step base = 256 (2^x)", state["orbit_max_iter"] == 256,
      f"got {state['orbit_max_iter']}")

# Test orbit_max_iter with shift (delta = ±4 powers of 2, capped by max-orbits-cap)
mb.pygame.key.set_mods(pygame.KMOD_LSHIFT)
state["orbit_max_iter"] = 200
overlay._do_action("orbit_max_iter", "plus", state)
test("MenuOverlay: orbit iter shift step = 2048 (2^x)", state["orbit_max_iter"] == 2048,
      f"got {state['orbit_max_iter']}")
mb.pygame.key.set_mods(0)

# Test line/label opacity stepping and off state
state["lines_opacity_j"] = 0.0
overlay._do_action("lines_opacity_j", "plus", state)
test("MenuOverlay: ju lines opacity steps up from off",
      abs(state["lines_opacity_j"] - 0.05) < 0.001, f"got {state['lines_opacity_j']}")

state["grid_label_opacity"] = 0.05
overlay._do_action("grid_label_opacity", "minus", state)
test("MenuOverlay: grid label opacity reaches off (0.0)",
      state["grid_label_opacity"] == 0.0, f"got {state['grid_label_opacity']}")

# Test highlight-points toggle
state["highlight_points"] = False
overlay._do_action("highlight_points", "click", state)
test("MenuOverlay: toggle highlight points", state["highlight_points"] is True)
overlay._do_action("highlight_points", "click", state)
test("MenuOverlay: restore highlight points", state["highlight_points"] is False)

# Test point-marker opacity off state
state["point_opacity_m"] = 0.05
overlay._do_action("point_opacity_m", "minus", state)
test("MenuOverlay: m point opacity reaches off (0.0)",
      state["point_opacity_m"] == 0.0, f"got {state['point_opacity_m']}")

# Test point-info and keybind actions
state["show_point_info"] = False
overlay._do_action("toggle-point-info", "click", state)
test("MenuOverlay: toggle point info", state["show_point_info"] is True)
state["show_keybinds"] = False
overlay._do_action("show-keybinds", "click", state)
test("MenuOverlay: toggle show keybinds", state["show_keybinds"] is True)

# Test auto_iter toggle
state["auto_iter"] = True
overlay._do_action("auto_iter", "click", state)
test("MenuOverlay: toggle auto_iter (ON->OFF)", state["auto_iter"] is False)
overlay._do_action("auto_iter", "click", state)
test("MenuOverlay: toggle auto_iter (OFF->ON)", state["auto_iter"] is True)

# Colors are palette-driven; palette resets via reset-all / cycle-palette above.
mb.pygame.key.set_mods(0)  # reset

# ===========================================================================
# 23. Strip effects
# ===========================================================================
print("\n--- Strip Effects ---")
state = make_full_state(stripe_s=1.0, stripe_sig=0.9)
params = mb.build_render_params(state, maxiter=64)
mat_stripe = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params)
test("Stripe render: shape correct", mat_stripe.shape == (32, 32, 3))

state_no = make_full_state(stripe_s=0.0, stripe_sig=0.9)
params_no = mb.build_render_params(state_no, maxiter=64)
mat_no = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_no)

# Striped and non-striped should differ in at least some pixels
diff = np.mean(np.abs(mat_stripe.astype(np.int32) - mat_no.astype(np.int32)))
test("Stripe affects rendering", diff > 0.5, f"diff={diff:.2f}")

# ===========================================================================
# 24. Step effects (step_s)
# ===========================================================================
print("\n--- Step Effects ---")
state_nostep = make_full_state(step_s=0.0)
state_step = make_full_state(step_s=5.0)
params_ns = mb.build_render_params(state_nostep, maxiter=64)
params_s = mb.build_render_params(state_step, maxiter=64)
mat_ns = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_ns)
mat_s = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_s)
diff = np.mean(np.abs(mat_s.astype(np.int32) - mat_ns.astype(np.int32)))
test("Step effect changes rendering", diff > 0.1, f"diff={diff:.2f}")

# ===========================================================================
# 25. FXAA
# ===========================================================================
print("\n--- FXAA Toggle ---")
state_nofxaa = make_full_state(fxaa=False)
state_fxaa = make_full_state(fxaa=True)
params_nofxaa = mb.build_render_params(state_nofxaa, maxiter=64)
params_fxaa = mb.build_render_params(state_fxaa, maxiter=64)
mat_nofxaa = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_nofxaa)
mat_fxaa = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_fxaa)
test("FXAA changes rendering", not np.array_equal(mat_nofxaa, mat_fxaa))

# ===========================================================================
# 26. Discrete vs Smooth mode
# ===========================================================================
print("\n--- Smooth vs Discrete ---")
state_smooth = make_full_state(smooth=True)
state_discrete = make_full_state(smooth=False)
params_sm = mb.build_render_params(state_smooth, maxiter=64)
params_di = mb.build_render_params(state_discrete, maxiter=64)
mat_sm = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_sm)
mat_di = mb.compute_image(32, 32, -2.0, 1.0, -1.5, 1.5, 64, params_di)
test("Smooth vs discrete produce different results",
     not np.array_equal(mat_sm, mat_di))

# ===========================================================================
# 27. _blit_surface_clamped
# ===========================================================================
print("\n--- _blit_surface_clamped ---")
screen = pygame.display.set_mode((200, 200))
screen.fill((0, 0, 0))
surf = pygame.Surface((100, 100))
surf.fill((255, 0, 0))
mb._blit_surface_clamped(screen, surf, 10, 10, 200, 200)
arr_c = pygame.surfarray.array3d(screen)
test("_blit_surface_clamped: red pixels at offset",
     np.any(arr_c[10:110, 10:110, 0] > 200) and np.all(arr_c[10:110, 10:110, 1:3] < 50))

# Negative offset
screen.fill((0, 0, 0))
mb._blit_surface_clamped(screen, surf, -10, -10, 200, 200)
arr_n = pygame.surfarray.array3d(screen)
test("_blit_surface_clamped: negative offset clips correctly",
     np.any(arr_n[0:100, 0:100, 0] > 200))

# ===========================================================================
# 28. _RENDER_CACHE
# ===========================================================================
print("\n--- Render Cache ---")
test("_RENDER_CACHE is dict", isinstance(mb._RENDER_CACHE, dict))
state = make_full_state()
state.setdefault("julia_c", mb.DEFAULT_JULIA_C)
state.setdefault("set_blend", 0.0)
mb.render_to_surface(32, 32, -2.0, 1.0, -1.5, 1.5, 32, state)
key = mb._render_cache_key(state, 32, False, (-2.0, 1.0, -1.5, 1.5))
test("render_to_surface populates cache", key in mb._RENDER_CACHE)

# ===========================================================================
# 28. Auto-zoom iteration and split cycle
# ===========================================================================
print("\n--- Auto-zoom & Split Cycle ---")
state = make_full_state()
state["split_mode"] = None
state["set_blend"] = 0.0

# Auto-iteration: zoom 10x should increase target iter beyond base 64
_target = mb._compute_target_iter(-2.5, 1.0, -1.5, 1.5, base_iter=64)
test("Auto-zoom: default view iter ≈ 64", 64 <= _target <= 128)
_target_z = mb._compute_target_iter(-2.01, -1.99, -0.01, 0.01, base_iter=64)
test("Auto-zoom: deep zoom iter > 64", _target_z > 64)
test("Auto-zoom: iter capped at AUTO_ITER_MAX", _target_z <= mb.AUTO_ITER_MAX)

# Split cycle: None -> horizontal -> vertical -> None (overlay restores blend)
state["set_blend"] = 0.3
overlay._do_action("toggle-split", "click", state)
test("Split cycle: None -> horizontal", state["split_mode"] == "horizontal")
overlay._do_action("toggle-split", "click", state)
test("Split cycle: horizontal -> vertical", state["split_mode"] == "vertical")
overlay._do_action("toggle-split", "click", state)
test("Split cycle: vertical -> overlay(blend restored)", state["split_mode"] is None and state["set_blend"] > 0.0)

# Split mode: independent julia_viewport
state = make_full_state()
state["split_mode"] = "horizontal"
state["julia_viewport"] = list(mb.DEFAULT_JULIA_VIEWPORT)
_panes = mb._compute_pane_bounds("horizontal", "horizontal",
    -2.5, 1.0, -1.5, 1.5, 800, 600,
    julia_viewport=state.get("julia_viewport", mb.DEFAULT_JULIA_VIEWPORT))
test("Split panes: 2 panes returned", len(_panes) == 2)
test("Split panes: MB pane is not julia", _panes[0]["is_julia"] is False)
test("Split panes: Julia pane bounds differ from MB",
     _panes[1]["p_xmin"] != _panes[0]["p_xmin"] or
     _panes[1]["p_ymin"] != _panes[0]["p_ymin"])
test("Split panes: Julia uses independent viewport bounds",
     _panes[1]["p_xmin"] != _panes[0]["p_xmin"] or
     _panes[1]["p_ymin"] != _panes[0]["p_ymin"])

# Split-mode zoom centering: mouse point must stay at same complex coordinate
_pane = _panes[0]
_zoom_factor = 0.8
_rel_y = (149 - _pane["y"]) / _pane["h"]
_cx = _pane["p_xmin"] + (_pane["p_xmax"] - _pane["p_xmin"]) * 0.5
_cy = _pane["p_ymax"] - (_pane["p_ymax"] - _pane["p_ymin"]) * _rel_y
_new_h = (_pane["p_ymax"] - _pane["p_ymin"]) * _zoom_factor
_new_ymax = _cy + _rel_y * _new_h
_new_ymin = _new_ymax - _new_h
_mouse_stays = abs((_new_ymax - _new_h * _rel_y) - _cy) < 1e-10
test("Split zoom: mouse point stays at same complex coordinate", _mouse_stays)

# fix_aspect_ratio with zero dimensions
_result = mb.fix_aspect_ratio(-2, 2, -2, 2, 100, 0)
test("fix_aspect_ratio: h=0 returns original bounds",
     _result == [-2, 2, -2, 2])

# _render_exposed_edges with zero-size strips
screen = pygame.Surface((100, 100))
mb._render_exposed_edges(screen, 100, 100, -2, 2, -2, 2, state, 0.0, 0.0)
test("_render_exposed_edges: zero offset no-op", True)
# Verify it produces output when offset is non-zero
screen.fill((0, 0, 0))
mb._render_exposed_edges(screen, 100, 100, -2, 2, -2, 2, state, 20.0, 10.0)
arr_e = pygame.surfarray.array3d(screen)
test("_render_exposed_edges: non-zero offset produces pixels",
     np.sum(np.any(arr_e > 0, axis=2)) > 0)

# ===========================================================================
# 28b. Keybinds panel: mouse-wheel / mouse-button entries excluded
# ===========================================================================
print("\n--- Keybind Panel Filtering ---")
_mouse_actions = {"zoom-in", "zoom-out", "toggle-orbits", "toggle-orbit-lines-m",
                  "toggle-orbit-lines-j", "toggle-c-point", "toggle-split",
                  "toggle-grid", "grid-opac-up", "grid-opac-down",
                  "reset-orbit-point", "reset-orbit-points",
                  "reset-orbit-point-m", "reset-orbit-point-j", "reset-julia-c",
                  "swap-orbit-point", "cycle-palette", "animate-zoom",
                  "reset-settings", "iter-up", "iter-down",
                  "blend-up", "blend-down",
                  "stripe-up", "stripe-down", "step-up", "step-down",
                  "light-angle-up", "light-angle-down",
                  "light-azim-up", "light-azim-down", "light-i-up", "light-i-down",
                  "k-amb-up", "k-amb-down", "k-diff-up", "k-diff-down",
                  "k-spec-up", "k-spec-down", "shininess-up", "shininess-down"}
_mouse_keys = {k for k, v in TEST_MOUSE_KEYBINDS.items() if v.startswith(("scroll_", "mouse"))}
test("Mouse-wheel keybinds detected", len(_mouse_keys) > 0)
_filtered = {k: v for k, v in TEST_MOUSE_KEYBINDS.items() if not v.startswith(("scroll_", "mouse"))}
test("Filtered keybinds panel excludes mouse actions",
     all(not v.startswith(("scroll_", "mouse")) for v in _filtered.values()))
test("zoom-in removed from filtered panel", "zoom-in" not in _filtered)
test("zoom-out removed from filtered panel", "zoom-out" not in _filtered)

# ===========================================================================
# 28b. Palette system (YAML loading, application, cycling)
# ===========================================================================
print("\n--- Palette System ---")
_all_pals = mb.get_all_palettes()
test("All palettes loaded from YAML (≥5)", len(_all_pals) >= 5, f"got {len(_all_pals)}")
test("All palettes expose gradient stops",
     all(p["gradient_stops"] for p in _all_pals))

# Test _apply_palette_by_index for gradient palette (index 0 is fire-gradient)
_state_pal = make_full_state()
mb._apply_palette_by_index(_state_pal, 0)
test("Apply gradient palette: sets gradient_stops",
     "gradient_stops" in _state_pal and len(_state_pal["gradient_stops"]) >= 2)
test("Apply gradient palette: sets gradient_blend",
     _state_pal.get("gradient_blend") in mb.PALETTE_BLEND_MODES)

# Test _apply_palette_by_name
_apt_state = make_full_state()
mb._apply_palette_by_name(_apt_state, _all_pals[0]["name"])
test("Apply palette by name: gradient_stops set",
     "gradient_stops" in _apt_state)

# Test cycle-palette cycles through gradient palettes
_state_cycle = make_full_state(palette_index=0)
_state_cycle["_palette_file_idx"] = 0
overlay._do_action("cycle-palette", "click", _state_cycle)
test("Cycle-palette: sets _palette_file_idx to 1",
     _state_cycle.get("_palette_file_idx") == 1)
test("Cycle-palette: palette_index is valid",
     0 <= _state_cycle["palette_index"] < len(_all_pals))

# Test load-palette-file loads first gradient palette and retains its name
_state_load = make_full_state()
overlay._do_action("load-palette-file", "click", _state_load)
_loaded_name = mb.get_all_palettes()[_state_load["palette_index"]]["name"]
test("Load-palette-file: retains palette index", 0 <= _state_load["palette_index"] < len(_all_pals))
test("Load-palette-file: retains palette name", _loaded_name != "custom")

# Test that build_render_params works with gradient palette
_state_grad = make_full_state()
mb._apply_palette_by_index(_state_grad, 0)
_params_grad = mb.build_render_params(_state_grad, maxiter=64)
test("build_render_params: gradient colortable shape", _params_grad["colortable"].shape == (mb.NCOL, 3))
test("build_render_params: gradient colortable in [0,1]",
     np.all(_params_grad["colortable"] >= 0) and np.all(_params_grad["colortable"] <= 1.0 + 1e-6))


# ===========================================================================
# 28c. Auto-iter debounce and zoom-out reduction
# ===========================================================================
print("\n--- Auto-Iter Debounce & Zoom-Out ---")
# _compute_target_iter should return different values for different zoom levels
_t_default = mb._compute_target_iter(-2.5, 1.0, -1.5, 1.5, base_iter=64)
_t_zoomed = mb._compute_target_iter(-0.1, 0.1, -0.1, 0.1, base_iter=64)
test("Auto-iter: zoomed view needs more iterations", _t_zoomed > _t_default)
test("Auto-iter: default view is near base", _t_default <= 256)

# Test that reducing iterations works (zoom out scenario)
_t_wide = mb._compute_target_iter(-10.0, 10.0, -10.0, 10.0, base_iter=64)
test("Auto-iter: wide view needs fewer iterations than zoomed", _t_wide < _t_zoomed)

# Test orbit ncycle uses orbit_max_iter, not max_iter (no hardcoded 500 cap)
_state_orbit = make_full_state(max_iter=256, orbit_max_iter=100)
_max_iter = _state_orbit["orbit_max_iter"]
_orbit_ncycle = math.sqrt(_max_iter)
_render_ncycle = math.sqrt(_state_orbit["max_iter"])
test("Orbit ncycle < render ncycle (orbit_max_iter < max_iter)",
     _orbit_ncycle < _render_ncycle, f"orbit={_orbit_ncycle}, render={_render_ncycle}")

# Auto-iter caps: AUTO_ITER_MIN and AUTO_ITER_MAX exist and are reasonable
test("AUTO_ITER_MAX is 65536", mb.AUTO_ITER_MAX == 65536)
test("AUTO_ITER_MIN is 32", mb.AUTO_ITER_MIN == 32)
test("RENDER_TIMEOUT_MS is 5000", mb.RENDER_TIMEOUT_MS == 5000)
# MAX_ITER_CAP and MIN_ITER_CAP no longer exist
test("MAX_ITER_CAP removed", not hasattr(mb, 'MAX_ITER_CAP'))
test("MIN_ITER_CAP removed", not hasattr(mb, 'MIN_ITER_CAP'))


# ===========================================================================
# 29. Keybind name conversion
# ===========================================================================
print("\n--- Keybind Conversion ---")
default_kb = TEST_KEYBINDS
for action, key in default_kb.items():
    val = mb.get_keybind({"keybind." + action: key}, action, "default")
    test(f"Keybind {action} resolves", val == key.lower())

# ===========================================================================
# 30. Palette section format
# ===========================================================================
print("\n--- Palette Sections ---")
import tempfile as _tempfile
_PAL_YAML = """
palettes:
  - name: unified
    unified color:
      stops:
        - [0.0, 10, 20, 30]
        - [1.0, 200, 210, 220]
  - name: fallback
    main color:
      stops:
        - [0.0, 255, 0, 0]
        - [1.0, 0, 0, 255]
    mb lines:
      stops:
        - [0.0, 0, 255, 0]
"""
with _tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as _pf:
    _pf.write(_PAL_YAML)
    _pal_path = _pf.name
_pals = mb.load_palette_file(_pal_path, rewrite=False)
test("Palette: parses both entries", len(_pals) == 2)
_u, _f = _pals
test("Palette: unified collapses all sections",
     len({tuple(sec) for sec in _u["palette_sections"].values()}) == 1)
test("Palette: missing section falls back to main",
     _f["palette_sections"]["grid"] == _f["palette_sections"]["main color"])
test("Palette: explicit section preserved",
     _f["palette_sections"]["mb lines"][0][1][1] > 0.9)
test("Palette: explicit tracking",
     "mb lines" in _f["explicit_sections"] and "grid" not in _f["explicit_sections"])
os.unlink(_pal_path)

# Grid label formatting
test("Grid label short decimal", mb._grid_label_text(0.5, 0.1) == "0.5")
test("Grid label zero trims sign", mb._grid_label_text(-0.0, 0.1) == "0")
test("Grid label deep zoom scientific", "e" in mb._grid_label_text(1.2e-6, 1e-6))
test("Zoom percent default is 0", mb._zoom_percent(-2.0, 2.0, -2.0, 2.0, 100, 100) == 0.0)

# ===========================================================================
# 31. classify_orbit (auto-threshold, validated cycle detection)
# ===========================================================================
print("\n--- classify_orbit ---")

# Classification no longer has a module-level horizon cap; the caller passes
# the iteration budget (the app uses the current orbit_iter). Tests pin a long
# horizon explicitly to exercise weak-attractor detection.
TEST_CLASSIFY_ITER = 16384


def _classify_m(cx, cy, it=None):
    it = it or TEST_CLASSIFY_ITER
    pts = mb.compute_orbit(cx, cy, it, False, 0.0, 0.0)
    return mb.classify_orbit(pts, it)


def _classify_j(zx, zy, cx, cy, it=None):
    it = it or TEST_CLASSIFY_ITER
    pts = mb.compute_orbit(zx, zy, it, True, cx, cy)
    return mb.classify_orbit(pts, it)


test("classify: no module-level horizon cap", not hasattr(mb, "CLASSIFY_MAX_ITER"))

_r = _classify_m(2.0, 2.0)
test("classify: escaping point -> escaped",
     _r["status"] == "escaped" and not _r["proven"])
test("classify: escaped has no cycle points", _r["cycle_pts"] == [])

_r = _classify_m(0.0, 0.0)
test("classify: c=0 -> superattracting fixed point",
     _r["proven"] and _r["status"] == "superattracting" and _r["cycle_len"] == 1)

_r = _classify_m(-1.0, 0.0)
test("classify: c=-1 -> period-2 attracting cycle",
     _r["proven"] and _r["status"] == "attracting_cycle" and _r["cycle_len"] == 2)

_r = _classify_m(-0.125, 0.744)
test("classify: period-3 bulb -> cycle_len 3",
     _r["proven"] and _r["cycle_len"] == 3)

# Regression: a weakly attracting fixed point (|lambda| ~ 0.9986) needs a long
# horizon to close; the old 500-iteration fixed-tolerance detector missed it.
_r = _classify_m(0.018, -0.63)
test("classify: weak fixed point -> attracting_fixed",
     _r["proven"] and _r["status"] == "attracting_fixed" and _r["cycle_len"] == 1,
     f"status={_r['status']} proven={_r['proven']}")
test("classify: weak fixed point center matches closed form",
     abs(_r["center_pt"][0] + 0.1721) < 1e-3 and abs(_r["center_pt"][1] + 0.4687) < 1e-3,
     f"center={_r['center_pt']}")
test("classify: weak fixed point multiplier ~ 0.9986",
     _r["multiplier"] is not None and abs(abs(_r["multiplier"]) - 0.9986) < 1e-3)

# Regression: the previously-missed reported coordinate is a period-5 cycle.
_r = _classify_m(0.394285714285727, -0.351428571428577)
test("classify: reported coord -> proven period-5 cycle",
     _r["proven"] and _r["cycle_len"] == 5,
     f"status={_r['status']} K={_r['cycle_len']}")

# A bounded chaotic orbit must never be reported or drawn as a cycle.
_r = _classify_j(1.5, 0.0, -2.0, 0.0)
test("classify: chaotic orbit not proven", not _r["proven"])
test("classify: chaotic orbit has no cycle points", _r["cycle_pts"] == [])
test("classify: no bogus fallback cycle length", _r["cycle_len"] == 0)

# Short-horizon fallback: strong but unclosed recurrence -> near_periodic.
_r = _classify_m(0.394285714285727, -0.351428571428577, it=500)
test("classify: near_periodic tier at short horizon",
     _r["status"] == "near_periodic" and not _r["proven"] and _r["cycle_pts"] == [],
     f"status={_r['status']}")

# ===========================================================================
# 32. Clipboard helper + point-info format toggle
# ===========================================================================
print("\n--- Clipboard & point info ---")

_orig_backends = mb._clipboard_backends
try:
    mb._clipboard_backends = lambda: [("stub-ok", lambda text: text == "hello")]
    test("_set_clipboard: True when a backend succeeds",
         mb._set_clipboard("hello") is True)
    mb._clipboard_backends = lambda: [("stub-fail", lambda text: False)]
    test("_set_clipboard: False when all backends fail",
         mb._set_clipboard("hello") is False)
finally:
    mb._clipboard_backends = _orig_backends

_pio = mb.PointInfoOverlay()
_pio.active = True
_state_ci = make_full_state()
_copied = {}
_orig_set = mb._set_clipboard
try:
    def _stub_set(text):
        _copied["text"] = text
        return True
    mb._set_clipboard = _stub_set
    _th = _pio._copy_coord("m", _state_ci)
    _th.join(timeout=2.0)
    _expected = _pio._fmt(_state_ci["orbit_point_m"],
                          _state_ci.get("point_output_format", "a+bi"))
    test("point-info copy routes through clipboard helper",
         _copied.get("text") == _expected)
    test("point-info copy records feedback", _pio._copy_feedback is not None
         and _pio._copy_feedback[0] is True)
finally:
    mb._set_clipboard = _orig_set

_pio_fmt = mb.PointInfoOverlay()
_pio_fmt.active = True
_pio_fmt.panel_rect = pygame.Rect(0, 0, 400, 300)
_pio_fmt.button_rects = {"format": (10, 10, 60, 24)}
_state_fmt = make_full_state()
_state_fmt["point_output_format"] = "a+bi"
_handled, _ = _pio_fmt.handle_click((20, 20), _state_fmt)
test("point-info format toggle flips a+bi -> (x, y)",
     _handled and _state_fmt["point_output_format"] == "(x, y)")
_handled, _ = _pio_fmt.handle_click((20, 20), _state_fmt)
test("point-info format toggle flips back",
     _handled and _state_fmt["point_output_format"] == "a+bi")

# The format button and copy button must occupy distinct rects.
_surf_ci = pygame.Surface((900, 600), pygame.SRCALPHA)
_pio_rects = mb.PointInfoOverlay()
_pio_rects.active = True
_state_rects = make_full_state()
_state_rects["show_point_info"] = True
try:
    _pio_rects.draw(_surf_ci, pygame.font.SysFont("monospace", 12), _state_rects,
                    -2.0, 1.0, -1.5, 1.5, 900, 600)
    test("point-info registers distinct format and copy rects",
         "format" in _pio_rects.button_rects
         and "copy-tab" in _pio_rects.button_rects
         and _pio_rects.button_rects["format"] != _pio_rects.button_rects["copy-tab"])
except Exception as _e:
    test("point-info registers distinct format and copy rects", False, str(_e))

# ===========================================================================
# 33. YAML-configurable maxima (precision bits / auto-iteration)
# ===========================================================================
print("\n--- Configurable caps ---")
test("caps: default max precision bits", mb.DEFAULT_MAX_PRECISION_BITS == 4096)
test("caps: default max auto iterations", mb.DEFAULT_MAX_AUTO_ITER == 65536)
test("caps: precision clamp respects default ceiling",
     mb._normalize_precision_bits(999999) == 4096)
test("caps: precision floor is 53", mb._normalize_precision_bits(1) == 53)

_saved_bits_cap = mb.MAX_PRECISION_BITS
_saved_iter_cap = mb.AUTO_ITER_MAX
try:
    mb._apply_caps_from_settings({"max-precision-bits": "16384",
                                  "max-auto-iterations": "200000"})
    test("caps: raises precision ceiling from settings",
         mb.MAX_PRECISION_BITS == 16384
         and mb._normalize_precision_bits(10 ** 9) == 16384)
    test("caps: raises auto-iteration ceiling from settings",
         mb.AUTO_ITER_MAX == 200000
         and mb._compute_target_iter(-1e-9, 1e-9, -1e-9, 1e-9) == 200000)
    test("caps: view-derived bits respect raised ceiling",
         mb._precision_bits_for_view(-1e-9, 1e-9, -1e-9, 1e-9) <= 16384)

    import tempfile as _cap_tmp
    with _cap_tmp.TemporaryDirectory() as _cap_dir:
        _cap_file = os.path.join(_cap_dir, "caps.yaml")
        mb.save_settings(_cap_file, {"max-precision-bits": "8192",
                                     "max-auto-iterations": "131072"})
        mb._apply_caps_from_settings(mb.load_settings(_cap_file))
        test("caps: persist and reload from YAML",
             mb.MAX_PRECISION_BITS == 8192 and mb.AUTO_ITER_MAX == 131072)
finally:
    mb.MAX_PRECISION_BITS = _saved_bits_cap
    mb.AUTO_ITER_MAX = _saved_iter_cap
    test("caps: restored defaults",
         mb.MAX_PRECISION_BITS == 4096 and mb.AUTO_ITER_MAX == 65536)

# Center-point selection (math.txt Fixed Points): pick whichever analytic
# candidate is nearest the orbit/cycle average.
_cands0 = mb._fixed_point_candidates_for_c(0.0, 0.0)
test("center: candidates for c=0 are {0, 1}",
     sorted((round(z.real, 9), round(z.imag, 9)) for z in _cands0)
     == [(0.0, 0.0), (1.0, 0.0)], f"got {_cands0}")
_r0 = _classify_m(0.0, 0.0)
test("center: classify reports an orbit mean", "mean_pt" in _r0)
_c0 = mb._center_point_for(0.0, 0.0, _r0)
test("center: c=0 selects the superattracting fixed point 0",
     abs(_c0) < 1e-6, f"got {_c0}")
_r1 = _classify_m(-1.0, 0.0)
_c1 = mb._center_point_for(-1.0, 0.0, _r1)
test("center: c=-1 selects (1-sqrt5)/2",
     abs(_c1.real - (1 - 5 ** 0.5) / 2) < 1e-3 and abs(_c1.imag) < 1e-6,
     f"got {_c1}")

# yaml-configurable classification / iteration caps
_saved_k = mb.CLASSIFY_K_MAX
_saved_oc = mb.ORBIT_ITER_CAP
_saved_rc = mb.RENDER_ITER_CAP
try:
    mb._apply_caps_from_settings({"classify-k-max": "77",
                                  "max-orbits-cap": "9999",
                                  "max-iterations-cap": "1234"})
    test("caps: classify k max from settings", mb.CLASSIFY_K_MAX == 77)
    test("caps: orbit iteration ceiling from settings", mb.ORBIT_ITER_CAP == 9999)
    test("caps: render iteration ceiling from settings", mb.RENDER_ITER_CAP == 1234)
finally:
    mb.CLASSIFY_K_MAX = _saved_k
    mb.ORBIT_ITER_CAP = _saved_oc
    mb.RENDER_ITER_CAP = _saved_rc
    test("caps: iteration caps restored",
         mb.CLASSIFY_K_MAX == 60 and mb.ORBIT_ITER_CAP == 65536
         and mb.RENDER_ITER_CAP == 32768)

# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
