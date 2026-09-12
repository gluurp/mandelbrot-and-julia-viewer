#!/usr/bin/env python3
"""Deep tool test suite for mandelbrot-testing-playground.py.

Covers deep zoom / perturbation theory, CLI/export,
menu/clipboard tools, and performance/profiling tools.
"""
import sys
import os
import math
import argparse
import tempfile
import importlib.util
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

_spec = importlib.util.spec_from_file_location(
    "mb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mandelbrot_testing_playground.py")
)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

import pygame
pygame.init()
pygame.display.set_mode((640, 480))

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


def make_state(**overrides):
    state = {
        "max_iter": 256,
        "rgb_thetas": list(mb.COLOR_THETAS[0]),
        "phase": 0.0,
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
        "show_orbits": False,
        "orbit_point": [0.018, -0.63],
        "orbit_max_iter": 200,
        "orbit_point_size": 3,
        "use_julia": False,
        "julia_c": [0.394, 0.338],
        "xmin": -2.5, "xmax": 1.0, "ymin": -1.5, "ymax": 1.5,
    }
    state.update(overrides)
    return state


def make_params(state, maxiter=64):
    params = mb.build_render_params(state, maxiter=maxiter)
    lt = mb.DEFAULT_LIGHT_ANGLE * 2 * math.pi
    la = mb.DEFAULT_LIGHT_AZIM * math.pi / 2
    li = mb.DEFAULT_LIGHT_I
    ka = mb.DEFAULT_K_AMBIANT
    kd = mb.DEFAULT_K_DIFFUSE
    ks = mb.DEFAULT_K_SPECULAR
    sh = mb.DEFAULT_SHININESS
    params["light"] = np.array([lt, la, li, ka, kd, ks, sh], dtype=np.float32)
    return params


# ── 1. Deep Zoom / Perturbation Theory ─────────────────────────────

print("=" * 60)
print("DEEP TOOL SUITE — Deep Zoom & Perturbation")
print("=" * 60)

print("\n--- needs_perturbation ---")
check("needs_perturbation: wide view False",
      mb.needs_perturbation(-2.0, 1.0, -1.5, 1.5) is False)
check("needs_perturbation: narrow True",
      mb.needs_perturbation(-2e-11, 2e-11, -1e-11, 1e-11) is True)
check("needs_perturbation: threshold exact False",
      mb.needs_perturbation(-5e-11, 5e-11, -5e-11, 5e-11) is False)
check("needs_perturbation: equal ranges True",
      mb.needs_perturbation(0.0, 1e-11, 0.0, 1e-11) is True)

print("\n--- compute_set_perturbed ---")
ref_orbit = mb.compute_orbit(0.018, -0.63, 50)
check("ref_orbit shape", ref_orbit.shape == (51, 2))
check("ref_orbit finite", np.all(np.isfinite(ref_orbit)))

ct = mb.make_colortable(np.array(mb.COLOR_THETAS[0], dtype=np.float64))
check("colortable shape", ct.shape == (4096, 3))

creal_arr = np.array([0.018, 0.0185, 0.019], dtype=np.float32)
cim_arr = np.array([-0.63, -0.631, -0.629], dtype=np.float32)
params = make_params(make_state(), maxiter=50)

try:
    result = mb.compute_set_perturbed(
        creal_arr, cim_arr, ref_orbit, 50, ct,
        ncycle=int(math.sqrt(50)),
        stripe_s=params["stripe_s"], stripe_sig=params["stripe_sig"],
        step_s=params["step_s"],
        diag=math.sqrt(0.018**2 + 0.63**2),
        light=params["light"], smooth=True)
    check("perturbed: shape", result.shape == (3, 3, 3))
    check("perturbed: finite", np.all(np.isfinite(result)))
    check("perturbed: dtype float64", result.dtype == np.float64)

    # Compare perturbed vs direct CPU computation (should be similar for non-glitch pixels)
    direct = mb.compute_set_cpu(
        creal_arr.astype(np.float64), cim_arr.astype(np.float64), 50,
        ct, int(math.sqrt(50)), 0.0, 0.9, 0.0,
        math.sqrt(0.018**2 + 0.63**2), params["light"], smooth=True)
    check("perturbed matches CPU within tolerance",
          np.allclose(result, direct, atol=0.05),
          f"max_diff={np.max(np.abs(result - direct))}")
except Exception as e:
    check(f"perturbed: no crash ({type(e).__name__})", False, str(e))

try:
    creal_n = np.array([0.0180001], dtype=np.float32)
    cim_n = np.array([-0.6300001], dtype=np.float32)
    ref_n = mb.compute_orbit(0.018, -0.63, 30)
    p_params = make_params(make_state(), maxiter=30)
    perturbed = mb.compute_set_perturbed(
        creal_n, cim_n, ref_n, 30, ct,
        ncycle=int(math.sqrt(30)),
        stripe_s=p_params["stripe_s"], stripe_sig=p_params["stripe_sig"],
        step_s=p_params["step_s"],
        diag=math.sqrt(0.018**2 + 0.63**2),
        light=p_params["light"], smooth=True)
    check("perturbed narrow: finite", np.all(np.isfinite(perturbed)))
except Exception as e:
    check(f"perturbed narrow: no crash ({type(e).__name__})", False, str(e))

try:
    creal_j = np.array([0.0], dtype=np.float32)
    cim_j = np.array([0.0], dtype=np.float32)
    ref_j = mb.compute_orbit(0.018, -0.63, 20, use_julia=True, julia_c_re=0.394, julia_c_im=0.338)
    j_params = make_params(make_state(use_julia=True), maxiter=20)
    result_j = mb.compute_set_perturbed(
        creal_j, cim_j, ref_j, 20, ct,
        ncycle=int(math.sqrt(20)),
        stripe_s=j_params["stripe_s"], stripe_sig=j_params["stripe_sig"],
        step_s=j_params["step_s"],
        diag=0.5,
        light=j_params["light"], smooth=True,
        use_julia=True, julia_c_re=0.394, julia_c_im=0.338)
    check("perturbed Julia: finite", np.all(np.isfinite(result_j)))
except Exception as e:
    check(f"perturbed Julia: no crash ({type(e).__name__})", False, str(e))


# ── 3. CLI / Export ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DEEP TOOL SUITE — CLI/Export")
print("=" * 60)

print("\n--- CLI Argument Parsing ---")
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=None, help="Max iterations")
parser.add_argument("--color", type=str, default=None,
                    help="Palette name: fire, ocean, viridis, etc.")
parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
parser.add_argument("--no-gpu", action="store_true", help="Force CPU rendering")
parser.add_argument("--julia", action="store_true", help="Render Julia set")
parser.add_argument("--blend", type=float, default=None,
                    help="Blend factor: 0=MB only, 1=Julia only")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output file path")
parser.add_argument("--size", "-s", type=str, default=None,
                    help="Window size WxH")
parser.add_argument("--center", type=str, default=None,
                    help="Center point: x,y")
parser.add_argument("--zoom", type=str, default=None, help="Zoom level")
parser.add_argument("--palette-file", type=str, default=None,
                    help="Custom palette YAML file")
parser.add_argument("--watch", action="store_true", default=None,
                    help="Watch for file changes")
parser.add_argument("--precision-mode", type=str, default=None,
                    help="Precision mode: auto, mpfr, float64, or perturbed")
parser.add_argument("--precision-bits", type=int, default=None,
                    help="MPFR precision in bits (53..4096)")
parser.add_argument("--no-uncertainty", action="store_true", default=None,
                    help="Do not mark bounded high-precision pixels as uncertain")

args = parser.parse_args(["--iter", "512", "--color", "fire", "--julia",
                           "--output", "/tmp/test.png", "--size", "800x600",
                           "--center", "0,0", "--zoom", "2.0"])
check("CLI: iter parsed", args.iter == 512)
check("CLI: color parsed", args.color == "fire")
check("CLI: julia flag", args.julia is True)
check("CLI: output path", args.output == "/tmp/test.png")
check("CLI: size parsed", args.size == "800x600")
check("CLI: center parsed", args.center == "0,0")
check("CLI: zoom parsed", args.zoom == "2.0")
check("CLI: gpu flag absent", args.gpu is False)
check("CLI: no-gpu flag absent", args.no_gpu is False)
check("CLI: blend default None", args.blend is None)
check("CLI: watch absent", args.watch is None)
check("CLI: precision mode absent", args.precision_mode is None)
check("CLI: precision bits absent", args.precision_bits is None)
check("CLI: no-uncertainty absent", args.no_uncertainty is None)
precision_args = parser.parse_args(["--precision-mode", "mpfr", "--precision-bits", "256", "--no-uncertainty"])
check("CLI: precision mode parsed", precision_args.precision_mode == "mpfr")
check("CLI: precision bits parsed", precision_args.precision_bits == 256)
check("CLI: no-uncertainty parsed", precision_args.no_uncertainty is True)

print("\n--- Precision helpers ---")
check("precision mode: high alias", mb._normalize_precision_mode("high") == "mpfr")
check("precision mode: invalid fallback", mb._normalize_precision_mode("unknown") == "auto")
check("precision bits: lower clamp", mb._normalize_precision_bits(32) == 53)
check("precision bits: upper clamp", mb._normalize_precision_bits(5000) == 4096)
resolved = mb._resolve_precision("auto", 4096, ("-0.7463", "-0.7462999999999999", "0.1102", "0.1102000000000001"))
check("precision auto: deep view resolved", resolved[0] == "auto" and 128 <= resolved[1] <= 4096)
expected_tier = "mpfr" if mb.gmpy2 is not None else "float64"
check("precision mpfr: backend selection",
      mb._select_precision_tier("-1e-20", "1e-20", "-1e-20", "1e-20", "mpfr") == expected_tier)
state = make_state(precision_mode="mpfr", precision_bits=256, show_uncertainty=False)
params = mb.build_render_params(state, maxiter=64,
                                view_bounds=("-0.7463", "-0.7462999999999999", "0.1102", "0.1102000000000001"))
check("precision params: normalized mode", params["precision_mode"] == "mpfr")
check("precision params: normalized bits", params["precision_bits"] == 256)
check("precision params: uncertainty", params["show_uncertainty"] is False)
base_bounds = ("-0.7463", "-0.7462999999999999", "0.1102", "0.1102000000000001")
key_a = mb._render_cache_key(state, 64, False, base_bounds)
key_b = mb._render_cache_key({**state, "precision_bits": 512}, 64, False, base_bounds)
key_c = mb._render_cache_key({**state, "show_uncertainty": True}, 64, False, base_bounds)
check("precision cache: bits change key", key_a != key_b)
check("precision cache: uncertainty changes key", key_a != key_c)

print("\n--- render_image_cli ---")
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "cli_test.png")
    cli_args = argparse.Namespace(
        iter=64, color=None, gpu=False, no_gpu=False, julia=False,
        blend=None, output=out_path, size=[128, 128], center=None,
        zoom=None, palette_file=None, watch=None, precision_mode=None,
        precision_bits=None, no_uncertainty=None,
    )
    try:
        result = mb.render_image_cli(cli_args)
        check("render_image_cli: file exists", os.path.exists(out_path))
        from PIL import Image
        img = Image.open(out_path)
        check("render_image_cli: valid PNG", img.size == (128, 128))
    except ImportError as e:
        if "imageio" in str(e) or "PIL" in str(e):
            check("render_image_cli: imageio/PIL unavailable (skipped)", True)
        else:
            check(f"render_image_cli: failed ({e})", False)
    except Exception as e:
        check(f"render_image_cli: failed ({type(e).__name__})", False, str(e))

print("\n--- Animation export ---")
with tempfile.TemporaryDirectory() as tmpdir:
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    gif_path = os.path.join(tmpdir, "test.gif")
    try:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, frames, fps=5)
        check("animation: GIF created", os.path.exists(gif_path))
        check("animation: GIF non-empty", os.path.getsize(gif_path) > 0)
    except ImportError:
        from PIL import Image
        for i, frame in enumerate(frames):
            Image.fromarray(frame, "RGB").save(
                os.path.join(tmpdir, f"frame_{i}.png"))
        check("animation: PNG frames (imageio unavailable)",
              sum(1 for f in os.listdir(tmpdir) if f.endswith('.png')) == len(frames))

# Re-init pygame (render_image_cli may have quit it)
try:
    pygame.display.set_mode((640, 480))
except Exception:
    pass


# ── 4. Menu / Clipboard Tools ────────────────────────────────────────

print("\n" + "=" * 60)
print("DEEP TOOL SUITE — Menu/Clipboard Tools")
print("=" * 60)

print("\n--- _format_complex ---")
formatted = mb.MenuOverlay._format_complex(0.394, 0.338, -2.5, 1.0, -1.5, 1.5)
check("_format_complex: returns string", isinstance(formatted, str) and len(formatted) > 0)
check("_format_complex: has real part", "0.39" in formatted or "0.394" in formatted)
check("_format_complex: has imag part", "0.34" in formatted or "0.338" in formatted)
check("_format_complex: has i suffix", "i" in formatted)

formatted_zoom = mb.MenuOverlay._format_complex(0.001, 0.002, -1e-6, 1e-6, -1e-6, 1e-6)
check("_format_complex: zoom shows precision",
      len(formatted_zoom) > len(formatted) or "." in formatted_zoom)

formatted_zero = mb.MenuOverlay._format_complex(0.0, 0.0, -2.5, 1.0, -1.5, 1.5)
check("_format_complex: zero handles", "0" in formatted_zero)

# _compute_zoom_percent
zoom_default = mb.MenuOverlay._compute_zoom_percent(-2.5, 1.0, -1.5, 1.5, 640, 480)
check("_compute_zoom_percent: default view near 0%", zoom_default < 5, f"got {zoom_default}")
zoom_deep = mb.MenuOverlay._compute_zoom_percent(-2e-10, 2e-10, -1e-10, 1e-10, 640, 480)
check("_compute_zoom_percent: deep zoom near 100%", zoom_deep > 90, f"got {zoom_deep}")
zoom_mid = mb.MenuOverlay._compute_zoom_percent(-0.5, 0.5, -0.5, 0.5, 640, 480)
check("_compute_zoom_percent: mid zoom in range", 5 < zoom_mid < 95, f"got {zoom_mid}")

print("\n--- MenuOverlay._do_action ---")
overlay = mb.MenuOverlay()
overlay.active = True
action_state = make_state()

try:
    original_iter = action_state["max_iter"]
    overlay._do_action("max_iter", "plus", action_state)
    check("_do_action max_iter plus: increases", action_state["max_iter"] > original_iter)

    overlay._do_action("max_iter", "minus", action_state)
    check("_do_action max_iter minus: back to original",
          action_state["max_iter"] == original_iter)
except Exception as e:
    check(f"_do_action max_iter: no crash ({type(e).__name__})", False, str(e))

try:
    original_hue = action_state["rgb_thetas"][0]
    overlay._do_action("hue_0", "plus", action_state)
    check("_do_action hue0 plus: changes", action_state["rgb_thetas"][0] != original_hue)
except Exception as e:
    check(f"_do_action hue0: no crash ({type(e).__name__})", False, str(e))

try:
    overlay._do_action("unknown_action", "click", action_state)
    check("_do_action: unknown no crash", True)
except Exception as e:
    check(f"_do_action unknown: crashed ({e})", False, str(e))

print("\n--- MenuOverlay.handle_click ---")
overlay2 = mb.MenuOverlay()
overlay2.active = True
result = overlay2.handle_click((10, 10), make_state())
check("handle_click: empty returns tuple", isinstance(result, tuple) and len(result) == 2)
check("handle_click: empty (False, False)", result == (False, False))

overlay3 = mb.MenuOverlay()
overlay3.active = False
result2 = overlay3.handle_click((10, 10), make_state())
check("handle_click: inactive (False, False)", result2 == (False, False))

print("\n--- _unpack_rects ---")
overlay4 = mb.MenuOverlay()
rect_info = (10, 20, 30, 40)
try:
    rects = overlay4._unpack_rects(rect_info, "test-btn")
    check("_unpack_rects: returns list", isinstance(rects, list))
except Exception as e:
    check(f"_unpack_rects: no crash ({type(e).__name__})", False, str(e))

print("\n--- Clipboard coord copy ---")
overlay5 = mb.MenuOverlay()
overlay5.active = True
coord_state = make_state()
try:
    overlay5._copy_coord("coord-copy-m", coord_state)
    check("_copy_coord: M no crash", True)
except Exception as e:
    check(f"_copy_coord M: crashed ({e})", False, str(e))

try:
    overlay5._copy_coord("coord-copy-j", coord_state)
    check("_copy_coord: J no crash", True)
except Exception as e:
    check(f"_copy_coord J: crashed ({e})", False, str(e))

try:
    overlay5._copy_coord("coord-copy-c", coord_state)
    check("_copy_coord: C no crash", True)
except Exception as e:
    check(f"_copy_coord C: crashed ({e})", False, str(e))

try:
    overlay5._copy_coord("coord-copy-nonexistent", coord_state)
    check("_copy_coord: invalid no crash", True)
except Exception as e:
    check(f"_copy_coord invalid: crashed ({e})", False, str(e))


# ── 5. Performance / Profiling Tools ─────────────────────────────────

print("\n" + "=" * 60)
print("DEEP TOOL SUITE — Performance/Profiling")
print("=" * 60)

print("\n--- PerfMonitor ---")
pm = mb.PerfMonitor(interval=0.1)
check("PerfMonitor: initial state", pm.running is False and pm.stats["fps"] == 0)

pm.start()
check("PerfMonitor: running after start", pm.running is True)
import time as _time
_time.sleep(0.3)
stats = pm.get_stats()
pm.stop()
check("PerfMonitor: stats is dict", isinstance(stats, dict))
check("PerfMonitor: required keys",
      all(k in stats for k in ["fps", "cpu", "gpu_util", "gpu_mem_used",
                                "gpu_mem_total", "mem_used", "mem_total"]))
check("PerfMonitor: stopped", pm.running is False)

print("\n--- RenderProfiler ---")
rp = mb.RenderProfiler()
check("RenderProfiler: initial avg 0", rp.get_avg_render_ms() == 0)
check("RenderProfiler: initial empty tiles",
      rp.get_stats()["tile_counts"] == {"gpu_tiles": 0, "cpu_tiles": 0})

rp.record_render(16.5, 4, 2)
rp.record_render(15.2, 4, 2)
rp.record_render(17.0, 3, 3)
check("RenderProfiler: avg computed",
      abs(rp.get_avg_render_ms() - (16.5 + 15.2 + 17.0) / 3) < 0.01)
s = rp.get_stats()
check("RenderProfiler: tile counts", s["tile_counts"] == {"gpu_tiles": 3, "cpu_tiles": 3})
check("RenderProfiler: max tile counts", s["max_tile_counts"] == {"gpu_tiles": 4, "cpu_tiles": 3})
check("RenderProfiler: distribution string", "GPU:3" in s["tile_distribution"] and "CPU:3" in s["tile_distribution"])

for i in range(70):
    rp.record_render(10.0 + i * 0.1, 1, 1)
check("RenderProfiler: history limited", len(rp.render_times) == 60)
check("RenderProfiler: has latest",
      abs(rp.render_times[-1] - (10.0 + 69 * 0.1)) < 0.01)


# ── 6. State Persistence / Settings ──────────────────────────────────

print("\n" + "=" * 60)
print("DEEP TOOL SUITE — Settings/Persistence")
print("=" * 60)

print("\n--- save/load settings ---")
with tempfile.TemporaryDirectory() as tmpdir:
    settings_file = os.path.join(tmpdir, "settings.yaml")
    test_settings = {
        "iteration-max": "512",
        "zoom-level": "3.5",
        "center-x": "0.123",
        "center-y": "0.456",
        "is-julia": "True",
        "julia-cx": "0.285",
        "julia-cy": "0.01",
        "palette-name": "fire",
    }
    mb.save_settings(settings_file, test_settings)
    check("save_settings: file created", os.path.exists(settings_file))
    loaded = mb.load_settings(settings_file)
    check("load_settings: returns dict", isinstance(loaded, dict))
    for k, v in test_settings.items():
        check(f"load_settings: {k}={v}", loaded.get(k) == v,
              f"got {loaded.get(k)}")

print("\n--- get_persistent_setting ---")
with tempfile.TemporaryDirectory() as tmpdir:
    settings_file = os.path.join(tmpdir, "persist.yaml")
    mb.save_settings(settings_file, {"iteration-max": "1024"})
    val = mb.get_persistent_setting({"iteration-max": "1024"}, "iteration-max", cast=int, default=256)
    check("get_persistent_setting: int", val == 1024)
    val_str = mb.get_persistent_setting({"iteration-max": "1024"}, "iteration-max", cast=str, default="256")
    check("get_persistent_setting: str", val_str == "1024")
    val_missing = mb.get_persistent_setting({}, "nonexistent", cast=int, default=999)
    check("get_persistent_setting: default", val_missing == 999)


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"DEEP TOOL RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
