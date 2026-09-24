#!/usr/bin/env python3
"""End-to-end stress test for mandelbrot_testing_playground.py.

Exercises run_render_mode with simulated key events, menu actions, grid
edge cases, and orbit drawing in various modes. Uses subprocess isolation
to catch crashes that unit tests miss.
"""
import os
import sys
import time
import subprocess
import tempfile

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("MB_PALETTE_REWRITE", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "mandelbrot_testing_playground.py")

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


def run_py(script, timeout=15):
    """Write a Python script to a temp file and run it. Returns (rc, stderr, stdout)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=HERE) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            cwd=HERE,
        )
        return proc.returncode, proc.stderr, proc.stdout
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def make_key_runner(keys):
    """Create a runner script that sends key events to run_render_mode."""
    return f'''
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys
import time
import threading
import pygame
import importlib.util

pygame.init()
pygame.display.set_mode((128, 128))
pygame.event.set_allowed(None)
pygame.event.set_allowed([pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.QUIT, pygame.VIDEORESIZE])

spec = importlib.util.spec_from_file_location("mb", {APP!r})
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

# Never let the simulated session write to the user's real settings file.
import atexit
import shutil
import tempfile
_tmp_settings = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
_tmp_settings.close()
try:
    shutil.copyfile(mb.SETTINGS_FILE, _tmp_settings.name)
except OSError:
    pass
mb.SETTINGS_FILE = _tmp_settings.name
atexit.register(lambda: os.path.exists(_tmp_settings.name) and os.unlink(_tmp_settings.name))

# The simulated session must not persist state at all.
mb._save_persistent_state = lambda *a, **k: (None, None)

keys = {keys!r}

def inject_events():
    time.sleep(0.3)
    for k in keys:
        key_const = getattr(pygame, k, None)
        if key_const is not None:
            ev = pygame.event.Event(pygame.KEYDOWN,
                                    unicode="", key=key_const,
                                    mod=0, state=[])
            pygame.event.post(ev)
    time.sleep(0.2)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

t = threading.Thread(target = inject_events)
t.daemon = True
t.start()

try:
    mb.run_render_mode({{}}, cli_iter=64)
    sys.exit(0)
except SystemExit:
    pass
except Exception as e:
    import traceback
    print(f"[CRASH] {{type(e).__name__}}: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''


def make_menu_runner():
    """Create a runner script that exercises MenuOverlay actions."""
    return f'''
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
pygame.display.set_mode((640, 480))
import importlib.util

spec = importlib.util.spec_from_file_location("mb", {APP!r})
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

state = {{
    "max_iter": 64, "gradient_stops": list(mb.DEFAULT_GRADIENT_STOPS), "gradient_blend": mb.DEFAULT_GRADIENT_BLEND,
    "use_gpu": False, "stripe_s": 0.0, "stripe_sig": 0.9, "step_s": 0.0,
    "light_angle": mb.DEFAULT_LIGHT_ANGLE, "light_azim": mb.DEFAULT_LIGHT_AZIM,
    "light_i": mb.DEFAULT_LIGHT_I, "k_ambiant": mb.DEFAULT_K_AMBIANT,
    "k_diffuse": mb.DEFAULT_K_DIFFUSE, "k_specular": mb.DEFAULT_K_SPECULAR,
    "shininess": mb.DEFAULT_SHININESS, "smooth": True, "fxaa": False,
    "palette_index": 0,
    "orbit_max_iter": 200, "julia_c": [0.394, 0.338],
    "grid_opacity": 0.4,
    "point_opacity_c": 1.0,
    "split_mode": None, "set_blend": 0.0,
    "orbit_point_m": [0.018, -0.63], "orbit_point_j": [0.0, 0.0],
    "julia_viewport": list(mb.DEFAULT_JULIA_VIEWPORT),
    "orbit_hover": False,
}}

actions = [
    "toggle-split", "smooth", "use_gpu", "fxaa",
    "highlight_points", "auto_iter", "toggle-point-info",
]

mb._RENDER_CACHE.clear()
overlay = mb.MenuOverlay()
overlay.active = True
pi = mb.PointInfoOverlay()
pi.active = True
font = pygame.font.SysFont("monospace", 16)
surf = pygame.Surface((640, 480))
keybinds = mb.load_keybinds(mb.load_settings(mb.SETTINGS_FILE))

for action_name in actions:
    try:
        mb.MenuOverlay()._do_action(action_name, "toggle", {{**state}})
        print(f"[OK] _do_action: {{action_name}}")
    except Exception as e:
        print(f"[ERR] _do_action: {{action_name}}: {{e}}")
        import traceback; traceback.print_exc()

for label in ["no-split", "split-h", "split-v"]:
    try:
        s = {{**state}}
        if label == "split-h":
            s["split_mode"] = "horizontal"
            s["set_blend"] = 0.0
        elif label == "split-v":
            s["split_mode"] = "vertical"
            s["set_blend"] = 0.0
        overlay.draw(surf, font, s, keybinds, -2.0, 1.0, -1.5, 1.5)
        pi.draw(surf, font, s, -2.0, 1.0, -1.5, 1.5)
        print(f"[OK] draw: {{label}}")
    except Exception as e:
        print(f"[ERR] draw: {{label}}: {{e}}")
        import traceback; traceback.print_exc()

pi.handle_click((100, 100), {{**state}})
overlay.handle_click((100, 100), {{**state}})
print("[OK] handle_click")
print("DONE")
'''


def make_grid_runner():
    """Create a runner script that tests _draw_grid edge cases."""
    return f'''
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
pygame.display.set_mode((128, 128))
import importlib.util

spec = importlib.util.spec_from_file_location("mb", {APP!r})
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

surface = pygame.Surface((128, 128))
grid_font = pygame.font.SysFont("monospace", 10)
state = {{"grid_opacity": 0.4}}

test_views = [
    (-2.0, 1.0, -1.5, 1.5),
    (-0.5, 0.5, -0.1, 0.1),
    (-0.001, 0.001, -0.001, 0.001),
    (-1e-10, 1e-10, -1e-10, 1e-10),
    (0.0, 0.0, 0.0, 0.0),
]

for i, (xmin, xmax, ymin, ymax) in enumerate(test_views):
    try:
        mb._draw_grid(surface, state, xmin, xmax, ymin, ymax, 128, 128, font=grid_font)
        print(f"[OK] grid view {{i}}")
    except Exception as e:
        print(f"[ERR] grid view {{i}}: {{e}}")

try:
    panes = mb._compute_pane_bounds("horizontal", "horizontal",
                                    -2.0, 2.0, -2.0, 2.0, 128, 128,
                                    julia_viewport=list(mb.DEFAULT_JULIA_VIEWPORT))
    for p in panes:
        mb._draw_grid(surface, state, p["p_xmin"], p["p_xmax"],
                      p["p_ymin"], p["p_ymax"], p["w"], p["h"],
                      origin_x=p["x"], origin_y=p["y"], font=grid_font)
    print("[OK] split-mode grid")
except Exception as e:
    print(f"[ERR] split-mode grid: {{e}}")
print("DONE")
'''


def make_orbit_runner():
    """Create a runner script that tests _draw_orbit across modes."""
    return f'''
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
pygame.init()
pygame.display.set_mode((200, 200))
import importlib.util

spec = importlib.util.spec_from_file_location("mb", {APP!r})
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)

mb.compute_orbit = lambda *a, **k: np.array([[0.0, 0.0]])
surface = pygame.Surface((200, 200))

states = [
    ({{"point_opacity_c": 1.0, "split_mode": None,
       "set_blend": 0.0, "cycle_opacity": 0.9, "center_opacity": 0.9}},
     "no-split blend-0"),
    ({{"point_opacity_c": 1.0, "split_mode": None,
       "set_blend": 0.5, "cycle_opacity": 0.0, "center_opacity": 0.0}}, "no-split blend-0.5"),
    ({{"point_opacity_c": 1.0, "split_mode": "horizontal",
       "cycle_opacity": 0.9, "center_opacity": 0.9}}, "split-h cycle+center"),
    ({{"point_opacity_c": 1.0, "split_mode": "vertical",
       "cycle_opacity": 0.0, "center_opacity": 0.0}}, "split-v no-cycle"),
]

for state, label in states:
    state.update({{
        "max_iter": 64, "gradient_stops": list(mb.DEFAULT_GRADIENT_STOPS), "gradient_blend": mb.DEFAULT_GRADIENT_BLEND,
        "use_gpu": False, "stripe_s": 0.0, "orbit_max_iter": 100,
        "orbit_point_m": [0.0, 0.0], "orbit_point_j": [0.0, 0.0],
        "julia_c": [0.394, 0.338], "julia_viewport": list(mb.DEFAULT_JULIA_VIEWPORT),
        "orbit_point_size": 3, "orbit_hover": False,
    }})
    state.setdefault("set_blend", 0.0)
    panes = None
    if state.get("split_mode"):
        panes = mb._compute_pane_bounds(state["split_mode"], "horizontal",
                                        -2.0, 2.0, -2.0, 2.0, 200, 200,
                                        julia_viewport=state["julia_viewport"])
    try:
        mb._draw_orbit(surface, state, -2.0, 2.0, -2.0, 2.0, 200, 200,
                       pane_bounds=panes)
        print(f"[OK] orbit: {{label}}")
    except Exception as e:
        print(f"[ERR] orbit: {{label}}: {{e}}")
        import traceback; traceback.print_exc()
print("DONE")
'''


def main():
    print("=" * 60)
    print("STRESS TEST: mandelbrot_testing_playground.py")
    print("=" * 60)

    # ── 1. Basic startup (with auto-quit via simulated events) ───────────
    print("\n--- Startup ---")
    rc, err, _ = run_py(make_key_runner([]), timeout=8)
    check("App starts and exits cleanly", rc == 0,
          f"rc={rc} err={err[:300]}")

    # ── 2. Key event integration ─────────────────────────────────────────
    print("\n--- Key event integration ---")
    test_cases = [
        ("Toggle menu overlay (m)",    ["K_m"]),
        ("Toggle orbit lines (l)",     ["K_l"]),
        ("Toggle smooth (u)",          ["K_u"]),
        ("Toggle FXAA (j)",            ["K_j"]),
        ("Toggle GPU (p)",             ["K_p"]),
        ("Toggle Julia (s)",           ["K_s"]),
        ("Toggle orbits (space)",      ["K_SPACE"]),
        ("Toggle grid (g)",            ["K_g"]),
        ("Toggle c-point (c)",         ["K_c"]),
        ("Toggle point info (h)",      ["K_h"]),
        ("Cycle palette (tab)",        ["K_TAB"]),
        ("Zoom in (equals)",           ["K_EQUALS"]),
        ("Zoom out (minus)",           ["K_MINUS"]),
        ("Reset view (o)",             ["K_o"]),
        ("Reset orbit M (i)",          ["K_i"]),
        ("Reset colors (backspace)",   ["K_BACKSPACE"]),
        ("Show keybinds (k)",          ["K_k"]),
        ("Toggle split (v)",           ["K_v"]),
        ("Menu then toggle GPU",       ["K_m", "K_p", "K_m"]),
        ("Toggle split then back",     ["K_v", "K_v"]),
        ("Full workflow",              ["K_m", "K_g", "K_c", "K_h", "K_l",
                                        "K_u", "K_s", "K_v", "K_p", "K_m"]),
    ]

    for name, keys in test_cases:
        rc, err, _ = run_py(make_key_runner(keys), timeout=8)
        check(f"Keys: {name}", rc == 0,
              f"rc={rc} err={err[:300]}")

    # ── 3. CLI headless render (--output flag) ───────────────────────────
    print("\n--- CLI headless render ---")
    for desc, args in [
        ("--iter 64",            ["--iter", "64", "--output", "cli_test_1.png"]),
        ("--julia",              ["--julia", "--iter", "64", "--output", "cli_test_2.png"]),
        ("--blend 0.5",          ["--blend", "0.5", "--iter", "64", "--output", "cli_test_3.png"]),
        ("--blend 1.0",          ["--blend", "1.0", "--iter", "64", "--output", "cli_test_4.png"]),
        ("--no-gpu",             ["--no-gpu", "--iter", "64", "--output", "cli_test_5.png"]),
    ]:
        env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
        out_file = os.path.join(HERE, "cli_test.png")
        out_name = f"stress_cli_{desc.replace(' ','_').replace('-', '_')}.png"
        out_path = os.path.join(HERE, out_name)
        args_fixed = list(args)
        if "cli_test" in args_fixed[-1]:
            args_fixed[-1] = out_path
        try:
            proc = subprocess.run(
                [sys.executable, APP] + args_fixed,
                capture_output=True, text=True, timeout=30,
                cwd=HERE, env=env,
            )
            ok = proc.returncode == 0 and os.path.exists(out_path)
            check(f"CLI {desc}", ok,
                  f"rc={proc.returncode} err={proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            check(f"CLI {desc}", False, "TIMEOUT")
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    # ── 4. Menu actions and overlay drawing ──────────────────────────────
    print("\n--- Menu actions and overlay drawing ---")
    rc, err, out = run_py(make_menu_runner(), timeout=15)
    check("Menu actions + drawing: no crash",
          "DONE" in out and "[ERR]" not in out,
          f"rc={rc} err={err[:300]} out={out[:300]}")

    # ── 5. Grid edge cases ───────────────────────────────────────────────
    print("\n--- Grid edge cases ---")
    rc, err, out = run_py(make_grid_runner(), timeout=15)
    check("Grid edge cases: no crash",
          "DONE" in out and "[ERR]" not in out,
          f"rc={rc} err={err[:300]} out={out[:300]}")

    # ── 6. Orbit drawing across modes ────────────────────────────────────
    print("\n--- Orbit drawing (all modes) ---")
    rc, err, out = run_py(make_orbit_runner(), timeout=15)
    check("Orbit drawing all modes: no crash",
          "DONE" in out and "[ERR]" not in out,
          f"rc={rc} err={err[:300]} out={out[:300]}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
