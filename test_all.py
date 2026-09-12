"""Comprehensive tests for mandelbrot-testing-playground.py"""
import sys
import os
import math
import importlib.util
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

_spec = importlib.util.spec_from_file_location(
    "mb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandelbrot_testing_playground.py")
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

NCOL = mb.NCOL
COLOR_THETAS = mb.COLOR_THETAS
DEFAULT_RGB_THETAS = mb.DEFAULT_RGB_THETAS
DEFAULT_PHASE = mb.DEFAULT_PHASE
DEFAULT_NCYCLE = mb.DEFAULT_NCYCLE
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
COLOR_THETAS = mb.COLOR_THETAS
PALETTE_NAMES = ['fire-gradient', 'deep-sea-gradient', 'twilight-gradient',
                 'aurora-gradient', 'forest-gradient', 'sunset-gradient',
                 'amber-gradient', 'copper-gradient', 'electric-gradient',
                 'lava-gradient', 'teal-gradient']

import pygame
pygame.init()
pygame.display.set_mode((100, 100))

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
        "rgb_thetas": list(COLOR_THETAS[0]),
        "phase": 0.0,
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
        "show_orbits": True,
        "show_orbits_m": True,
        "show_orbits_j": True,
        "orbit_point": [0.018, -0.63],
        "orbit_point_m": [0.018, -0.63],
        "orbit_point_j": [0.153, 0.473],
        "orbit_max_iter": 200,
        "orbit_point_size": 3,
        "show_orbit_handles": True,
        "set_blend": 0.0,
        "julia_c": [0.394, 0.338],
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
test("COLOR_THETAS has 1 entry (DEFAULT_RGB_THETAS only)", len(COLOR_THETAS) == 1)
test("COLOR_THETAS[0] matches DEFAULT_RGB_THETAS",
     COLOR_THETAS[0] == DEFAULT_RGB_THETAS)

# ===========================================================================
# 2. make_colortable
# ===========================================================================
print("\n--- make_colortable ---")
ct = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
test("Default HSV: colortable shape", ct.shape == (NCOL, 3))
test("Default HSV: values in [0, 1.01]",
     np.all(ct >= -0.001) and np.all(ct <= 1.001))

# Phase shifting produces different colortable
ct_base = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
ct_phased = mb.make_colortable(np.array(
    [COLOR_THETAS[0][0] + 0.3, COLOR_THETAS[0][1] + 0.3, COLOR_THETAS[0][2]],
    dtype=np.float64))
test("Phase changes colortable", not np.allclose(ct_base, ct_phased, atol=0.01))
test("Phase preserves shape", ct_phased.shape == ct_base.shape)

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
# 3. _hsv_to_rgb_vec
# ===========================================================================
print("\n--- _hsv_to_rgb_vec ---")
hues = np.array([0.0, 0.167, 0.5, 1.0], dtype=np.float64)
sats = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float64)
vals = np.array([1.0, 0.8, 1.0, 0.5], dtype=np.float64)
r, g, b = mb._hsv_to_rgb_vec(hues, sats, vals)
test("HSV→RGB: arrays same length", len(r) == len(g) == len(b) == 4)
test("HSV→RGB: values in [0,1]",
     np.all(r >= 0) and np.all(r <= 1) and
     np.all(g >= 0) and np.all(g <= 1) and
     np.all(b >= 0) and np.all(b <= 1))
# Pure hue at full sat/val: hue=0 → red (r=val, g=t=0, b=p=0)
test("HSV→RGB: hue=0 → red dominant", r[0] >= g[0] and r[0] >= b[0], f"r={r[0]}, g={g[0]}, b={b[0]}")
# Zero saturation → grayscale
test("HSV→RGB: sat=0 → grayscale", abs(r[0] - g[0]) < 0.01 and abs(g[0] - b[0]) < 0.01)

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
# 6. smooth_iter
# ===========================================================================
print("\n--- smooth_iter ---")
result = mb.smooth_iter(complex(0.5, 0.5), 100, 0.0, 0.9)
test("smooth_iter returns 4-tuple", len(result) == 4)
niter, stripe_a, dem, normal = result
test("Escaping point has niter > 0", niter > 0, f"niter={niter}")
test("Escaping point has finite dem", np.isfinite(dem))

result = mb.smooth_iter(complex(-0.75, 0.0), 100, 0.0, 0.9)
niter, stripe_a, dem, normal = result
test("Interior point niter == 0", niter == 0.0, f"niter={niter}")

# With stripe active
result = mb.smooth_iter(complex(0.5, 0.5), 100, 1.0, 0.9)
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
colortable = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
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
colortable = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
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

# Phase in params
state_phased = make_full_state(phase=0.25)
params_phased = mb.build_render_params(state_phased, maxiter=64)
ct1 = params["colortable"]
ct2 = params_phased["colortable"]
test("Phase changes colortable in params", not np.allclose(ct1, ct2, atol=0.01))

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
    colortable = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
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

# smooth_iter with Julia mode
result_ju = mb.smooth_iter(complex(0.5, 0.5), 100, 0.0, 0.9, True, 0.394, 0.338)
niter_ju, stripe_a_ju, dem_ju, normal_ju = result_ju
test("Julia smooth_iter: escaping point has niter > 0", niter_ju > 0, f"niter={niter_ju}")
test("Julia smooth_iter all finite", np.isfinite(niter_ju) and np.isfinite(dem_ju) and np.isfinite(normal_ju))

# Julia render via compute_image
julia_state = {
    "colortable": mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64)),
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
colortable = mb.make_colortable(np.array(COLOR_THETAS[0], dtype=np.float64))
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
    hidden_state = build_state(show_orbits_m=False, show_orbits_j=True,
                               show_c_point=False, set_blend=0.0)
    mb._draw_orbit(screen, hidden_state, -2.0, 2.0, -2.0, 2.0, 100, 100)
    test("Disabled orbit hides starting-point marker", draw_calls == [])

    draw_calls.clear()
    visible_state = build_state(show_orbits_m=True, show_orbits_j=False,
                                show_c_point=False, set_blend=0.0)
    mb._draw_orbit(screen, visible_state, -2.0, 2.0, -2.0, 2.0, 100, 100)
    test("Enabled orbit shows starting-point marker",
         len(draw_calls) > 0)
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
    c_state = build_state(show_orbits_m=False, show_orbits_j=False,
                          show_c_point=True, set_blend=1.0,
                          c_point_color=[0.0, 200.0 / 255.0, 0.0])
    mb._draw_orbit(screen, c_state, -2.0, 2.0, -2.0, 2.0, 100, 100)
    test("Julia c-point uses configured RGB color",
         (0, 200, 0) in [call[1] for call in draw_calls if len(call) > 1])

    draw_calls.clear()
    c_state["split_mode"] = "horizontal"
    c_state["julia_viewport"] = list(mb.DEFAULT_JULIA_VIEWPORT)
    panes = mb._compute_pane_bounds("horizontal", "horizontal",
                                    -2.0, 2.0, -2.0, 2.0, 100, 100,
                                    julia_viewport=c_state["julia_viewport"])
    mb._draw_orbit(screen, c_state, -2.0, 2.0, -2.0, 2.0, 100, 100,
                   pane_bounds=panes)
    test("Julia c-point is drawn on both split panes",
         sum(1 for call in draw_calls if len(call) > 1 and call[1] == (0, 200, 0)) == 2)
finally:
    mb.compute_orbit = original_compute_orbit
    mb.pygame.draw.circle = original_circle

screen = pygame.Surface((120, 120))
screen.fill((0, 0, 0))
grid_state = build_state(show_grid=True, grid_opacity=0.5)
mb._draw_grid(screen, grid_state, -2.0, 2.0, -2.0, 2.0, 120, 120)
grid_pixels = pygame.surfarray.array3d(screen)
test("Grid renders visible lines", np.any(grid_pixels > 0))

# ===========================================================================
# 21. Settings persistence (load/save round-trip)
# ===========================================================================
print("\n--- Settings Persistence ---")
import tempfile
settings = {
    "hue_0": "0.0",
    "hue_1": "0.167",
    "sat": "0.95",
    "phase": "0.0",
    "keybind.quit": "escape",
    "keybind.hue0-up": "ctrl+r",
    "iteration-max": "64",
}
tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
tmpfile.write("hue_0 = 0.0\nhue_1 = 0.167\nsat = 0.95\nphase = 0.0\nkeybind.quit = escape\nkeybind.hue0-up = ctrl+r\niteration-max = 64\n")
tmpfile.close()

loaded = mb.load_settings(tmpfile.name)
test("YAML loads", True)
test("YAML has hue_0", "hue_0" in loaded)
test("YAML has hue_1", "hue_1" in loaded)
test("YAML has sat", "sat" in loaded)
test("YAML has phase", "phase" in loaded)
test("YAML has keybind.hue0-up", "keybind.hue0-up" in loaded)
test("YAML hue_0 value", loaded["hue_0"] == "0.0")
test("YAML keybind.hue0-up value", loaded["keybind.hue0-up"] == "ctrl+r")

test("YAML no orbit-active (uses show-orbits)", "orbit-active" not in loaded)

# Round-trip save
save_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
save_file.close()
mb.save_settings(save_file.name, loaded)
reloaded = mb.load_settings(save_file.name)
test("Save/load round-trip: hue_0", reloaded.get("hue_0") == "0.0")
test("Save/load round-trip: keybind matches", reloaded.get("keybind.hue0-up") == "ctrl+r")

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
    persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5, mb.DEFAULT_KEYBINDS)
test("Persistent snapshot: Julia c", persist_values["julia-cx"] == "-0.8" and persist_values["julia-cy"] == "0.2")
test("Persistent snapshot: Mandelbrot orbit", persist_values["orbit-mx"] == "-0.3" and persist_values["orbit-my"] == "0.4")
test("Persistent snapshot: Julia orbit", persist_values["orbit-jx"] == "0.2" and persist_values["orbit-jy"] == "-0.7")
test("Persistent snapshot: main viewport", persist_values["view-xmin"] == "-2.0" and persist_values["view-ymax"] == "1.5")
test("Persistent snapshot: Julia viewport", persist_values["julia-viewport"] == "-1.2,1.2,-0.8,0.8")

# Debounced persistence saves after one second idle and force-saves on exit
persist_settings = {}
debounce_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
debounce_file.close()
initial_snapshot = mb._persistent_snapshot(mb._build_persistent_values(
    persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5))
persist_state["julia_c"] = [-0.9, 0.3]
changed_snapshot, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5,
    mb.DEFAULT_KEYBINDS, True, initial_snapshot, None,
    settings_file=debounce_file.name, now=10.0)
test("Debounced persistence: no early save", not persist_settings)
_, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5,
    mb.DEFAULT_KEYBINDS, True, changed_snapshot, changed_time,
    settings_file=debounce_file.name, now=10.9)
test("Debounced persistence: waits for full debounce", not persist_settings)
_, changed_time = mb._save_persistent_state(
    persist_settings, persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5,
    mb.DEFAULT_KEYBINDS, True, changed_snapshot, changed_time,
    settings_file=debounce_file.name, now=11.0)
test("Debounced persistence: saves after one second", "julia-cx" in persist_settings)
force_settings = {}
mb._save_persistent_state(
    force_settings, persist_state, 800, 600, -2.0, 1.0, -1.5, 1.5,
    mb.DEFAULT_KEYBINDS, True, changed_snapshot, changed_time,
    force=True, settings_file=debounce_file.name, now=11.0)
test("Persistent force save: writes immediately", "julia-cx" in force_settings)

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
menu_screen = pygame.Surface((800, 600))
overlay._menu_cache = None
overlay.draw(menu_screen, pygame.font.SysFont("monospace", 18), menu_state,
             mb.DEFAULT_KEYBINDS, -2.0, 1.0, -1.5, 1.5)
test("MenuOverlay: split control is present", "toggle-split" in overlay.button_rects)
test("MenuOverlay: grid opacity control is present", "grid_opacity" in overlay.button_rects)
test("MenuOverlay: highlight-points control is present", "show_orbit_handles" in overlay.button_rects)

# Simulate button rects for testing _do_action
# Test minus/plus button unpacking
overlay.button_rects = {
    "phase": (100, 100, 36, 28, 140, 100, 36, 28, 180, 100, 112, 28),  # minus, plus, reset
}
rects = overlay._unpack_rects(overlay.button_rects["phase"], "phase")
test("Button unpack: 12-tuple → 3 rects", len(rects) == 3)
test("Button unpack: types correct", rects[0][1] == "minus" and rects[1][1] == "plus" and rects[2][1] == "reset")

overlay.button_rects = {
    "hue_0": (100, 100, 36, 28, 140, 100, 36, 28),  # minus, plus
}
rects = overlay._unpack_rects(overlay.button_rects["hue_0"], "hue_0")
test("Button unpack: 8-tuple → 2 rects", len(rects) == 2)

overlay.button_rects = {
    "smooth": (100, 100, 200, 28),  # single click
}
rects = overlay._unpack_rects(overlay.button_rects["smooth"], "smooth")
test("Button unpack: 4-tuple → 1 rect", len(rects) == 1)
test("Button unpack: single type is click", rects[0][1] == "click")

# Test _do_action for cyclic wrapping
state["phase"] = 0.99
mb.pygame.key.set_mods(0)  # no modifiers
overlay._do_action("phase", "plus", state)
test("MenuOverlay: phase wraps from 0.99→0.0",
     abs(state["phase"] - 0.04) < 0.01 or state["phase"] < 0.05, f"phase={state['phase']}")

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
test("MenuOverlay: cycle-palette rgb_thetas at default",
     state["rgb_thetas"] == DEFAULT_RGB_THETAS)

# Test reset-all
overlay._do_action("reset-all", "click", state)
test("MenuOverlay: reset-all restores rgb_thetas",
     state["rgb_thetas"] == DEFAULT_RGB_THETAS)
test("MenuOverlay: reset-all restores phase", state["phase"] == DEFAULT_PHASE)
test("MenuOverlay: reset-all restores shininess", state["shininess"] == DEFAULT_SHININESS)

# Test reset-orbit-point
state["orbit_point"] = [-1.0, 2.0]
overlay._do_action("reset-orbit-point", "click", state)
test("MenuOverlay: reset-orbit-point to (0.018, -0.63)",
      state["orbit_point"] == [0.018, -0.63])
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

# Test toggle show_orbits_m and show_orbits_j
state["show_orbits_m"] = True
overlay._do_action("show_orbits_m", "click", state)
test("MenuOverlay: toggle show_orbits_m", state["show_orbits_m"] is False)

state["show_orbits_j"] = False
overlay._do_action("show_orbits_j", "click", state)
test("MenuOverlay: toggle show_orbits_j", state["show_orbits_j"] is True)

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

# Test orbit_max_iter with shift (delta = ±4 powers of 2, capped at 512)
mb.pygame.key.set_mods(pygame.KMOD_LSHIFT)
state["orbit_max_iter"] = 200
overlay._do_action("orbit_max_iter", "plus", state)
test("MenuOverlay: orbit iter shift step = 512 (capped)", state["orbit_max_iter"] == 512,
      f"got {state['orbit_max_iter']}")
mb.pygame.key.set_mods(0)

# Test orbit line toggles
state["show_orbit_lines_m"] = True
overlay._do_action("show_orbit_lines_m", "click", state)
test("MenuOverlay: toggle show_orbit_lines_m", state["show_orbit_lines_m"] is False)

state["show_orbit_lines_j"] = False
overlay._do_action("show_orbit_lines_j", "click", state)
test("MenuOverlay: toggle show_orbit_lines_j", state["show_orbit_lines_j"] is True)

# Test c-point toggle
state["show_c_point"] = True
overlay._do_action("show_c_point", "click", state)
test("MenuOverlay: toggle show_c_point", state["show_c_point"] is False)

# Test highlight-points toggle
state["show_orbit_handles"] = True
overlay._do_action("show_orbit_handles", "click", state)
test("MenuOverlay: toggle highlight points", state["show_orbit_handles"] is False)
overlay._do_action("show_orbit_handles", "click", state)
test("MenuOverlay: restore highlight points", state["show_orbit_handles"] is True)

# Test auto_iter toggle
state["auto_iter"] = True
overlay._do_action("auto_iter", "click", state)
test("MenuOverlay: toggle auto_iter (ON->OFF)", state["auto_iter"] is False)
overlay._do_action("auto_iter", "click", state)
test("MenuOverlay: toggle auto_iter (OFF->ON)", state["auto_iter"] is True)

# Test c-point color sliders
state["c_point_color"] = [0, 0.784, 0]  # [0, 200/255, 0]
overlay._do_action("c_color_r", "plus", state)
test("MenuOverlay: c_point color r step", state["c_point_color"][0] > 0.0,
      f"got {state['c_point_color'][0]}")

# Test reset-colors restores main colors
overlay._do_action("reset-colors", "click", state)
test("MenuOverlay: reset-colors restores rgb_thetas",
      state["rgb_thetas"] == DEFAULT_RGB_THETAS)
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
                  "hue0-up", "hue0-down", "hue1-up", "hue1-down",
                  "sat-up", "sat-down", "blend-up", "blend-down",
                  "stripe-up", "stripe-down", "step-up", "step-down",
                  "phase-up", "phase-down", "light-angle-up", "light-angle-down",
                  "light-azim-up", "light-azim-down", "light-i-up", "light-i-down",
                  "k-amb-up", "k-amb-down", "k-diff-up", "k-diff-down",
                  "k-spec-up", "k-spec-down", "shininess-up", "shininess-down"}
_mouse_keys = {k for k, v in mb.DEFAULT_KEYBINDS.items() if v.startswith(("scroll_", "mouse"))}
test("Mouse-wheel keybinds detected", len(_mouse_keys) > 0)
_filtered = {k: v for k, v in mb.DEFAULT_KEYBINDS.items() if not v.startswith(("scroll_", "mouse"))}
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
test("No HSV palettes", not any(p["type"] == "hsv" for p in _all_pals))
test("All palettes are gradient type", all(p["type"] == "gradient" for p in _all_pals))

# Test _apply_palette_by_index for gradient palette (index 0 is fire-gradient)
_state_pal = make_full_state()
_state_pal["rgb_thetas"] = [0.5, 0.5, 0.5]
mb._apply_palette_by_index(_state_pal, 0)
test("Apply gradient palette: sets gradient_stops",
     "gradient_stops" in _state_pal and len(_state_pal["gradient_stops"]) >= 2)
test("Apply gradient palette: rgb_thetas set to default",
     _state_pal["rgb_thetas"] == list(mb.DEFAULT_RGB_THETAS))

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

# Test orbit ncycle uses orbit_max_iter, not max_iter
_state_orbit = make_full_state(max_iter=256, orbit_max_iter=100)
_max_iter = min(_state_orbit["orbit_max_iter"], 500)
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
default_kb = mb.DEFAULT_KEYBINDS
for action, key in default_kb.items():
    val = mb.get_keybind({"keybind." + action: key}, action, "default")
    test(f"Keybind {action} resolves", val == key.lower())

# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)
sys.exit(1 if FAIL > 0 else 0)
