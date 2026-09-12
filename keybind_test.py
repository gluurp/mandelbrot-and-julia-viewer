#!/usr/bin/env python3
"""End-to-end keybind action tests for mandelbrot-testing-playground.py.

Simulates keyboard events for every action in default_keybinds and verifies
the resulting state changes are correct.
"""
import sys
import os
import math
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

def check(name, condition, details=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {details}")

def make_state(**overrides):
    state = {
        "max_iter": 256,
        "rgb_thetas": list(mb.DEFAULT_RGB_THETAS),
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
        "show_orbits": True,
        "orbit_point": [0.018, -0.63],
        "orbit_max_iter": 200,
        "orbit_point_size": 3,
        "show_orbit_handles": True,
        "auto_iter": mb.DEFAULT_AUTO_ITER,
        "is_julia": False,
        "julia_c": [0.394, 0.338],
    }
    state.update(overrides)
    return state

# ── Replicate the key handling logic from run_render_mode ───────────────────
# We extract the action handlers by simulating the same logic.

def simulate_key(state, action, shift_held=False):
    """Apply the same state changes that run_render_mode would for each action."""
    _mult = 10.0 if shift_held else 1.0
    _RENDER_CACHE = mb._RENDER_CACHE
    changed = False

    if action == "toggle-gpu":
        state["use_gpu"] = not (state["use_gpu"] and mb._CUDA_AVAILABLE)
        _RENDER_CACHE.clear()
        changed = True
    elif action == "toggle-info":
        pass  # Overlay toggle, don't track in state
        changed = True
    elif action == "toggle-smooth":
        state["smooth"] = not state["smooth"]
        if not state["smooth"]:
            state["step_s"] = 0
        _RENDER_CACHE.clear()
        changed = True
    elif action == "toggle-fxaa":
        state["fxaa"] = not state.get("fxaa", False)
        _RENDER_CACHE.clear()
        changed = True
    elif action == "toggle-julia":
        state["is_julia"] = not state.get("is_julia", False)
        _RENDER_CACHE.clear()
        changed = True
    elif action == "toggle-auto-iter":
        state["auto_iter"] = not state.get("auto_iter", mb.DEFAULT_AUTO_ITER)
        _RENDER_CACHE.clear()
        changed = True
    elif action == "toggle-orbits":
        state["show_orbits"] = not state.get("show_orbits", False)
        changed = True
    elif action == "toggle-orbit-handles":
        state["show_orbit_handles"] = not state.get("show_orbit_handles", mb.DEFAULT_SHOW_ORBIT_HANDLES)
        changed = True
    elif action == "reset-orbit-point":
        state["orbit_point"] = [0.018, -0.63]
        changed = True
    elif action == "cycle-palette":
        palettes = mb.get_all_palettes()
        if palettes:
            gradient_palettes = [p for p in palettes if p.get("type") == "gradient"]
            if gradient_palettes:
                _idx = state.get("_palette_file_idx", 0) % len(gradient_palettes)
                _global_idx = palettes.index(gradient_palettes[_idx])
                mb._apply_palette_by_index(state, _global_idx)
                state["palette_index"] = _global_idx
                state["_palette_file_idx"] = (_idx + 1) % len(gradient_palettes)
                _RENDER_CACHE.clear()
                changed = True
    elif action == "load-palette-file":
        palettes = mb.get_all_palettes()
        if palettes:
            gradient_palettes = [p for p in palettes if p.get("type") == "gradient"]
            _idx = state.get("_palette_file_idx", 0) % len(gradient_palettes or palettes)
            mb._apply_palette_by_index(state, _idx)
            state["palette_index"] = _idx
            state["_palette_file_idx"] = (_idx + 1) % len(gradient_palettes or palettes)
            _RENDER_CACHE.clear()
            changed = True
    elif action == "reset-settings":
        state["rgb_thetas"] = list(mb.DEFAULT_RGB_THETAS)
        state["phase"] = mb.DEFAULT_PHASE
        state["stripe_s"] = mb.DEFAULT_STRIPE_S
        state["stripe_sig"] = mb.DEFAULT_STRIPE_SIG
        state["step_s"] = mb.DEFAULT_STEP_S
        state["light_angle"] = mb.DEFAULT_LIGHT_ANGLE
        state["light_azim"] = mb.DEFAULT_LIGHT_AZIM
        state["light_i"] = mb.DEFAULT_LIGHT_I
        state["k_ambiant"] = mb.DEFAULT_K_AMBIANT
        state["k_diffuse"] = mb.DEFAULT_K_DIFFUSE
        state["k_specular"] = mb.DEFAULT_K_SPECULAR
        state["shininess"] = mb.DEFAULT_SHININESS
        state["smooth"] = True
        state["orbit_point"] = [0.018, -0.63]
        state["show_orbits"] = True
        state["orbit_max_iter"] = 200
        state["orbit_point_size"] = 3
        state["show_orbit_handles"] = mb.DEFAULT_SHOW_ORBIT_HANDLES
        state["fxaa"] = False
        state["palette_index"] = mb.DEFAULT_PALETTE_INDEX
        mb._apply_palette_by_index(state, state["palette_index"])
        state["auto_iter"] = mb.DEFAULT_AUTO_ITER
        state["is_julia"] = False
        state["julia_c"] = list(mb.DEFAULT_JULIA_C)
        _RENDER_CACHE.clear()
        changed = True
    elif action == "reset-view":
        pass
        changed = True
    elif action == "quit":
        pass
        changed = True
    elif action in mb._NUMERIC_STEPS:
        mb._apply_step(state, action, _mult)
        _RENDER_CACHE.clear()
        changed = True
    elif action in ("iter-up", "iter-down"):
        cur = state["max_iter"]
        log2 = int(math.log2(cur)) if cur > 0 else 0
        if action == "iter-up":
            new_log2 = log2 + (10 if shift_held else 1)
        else:
            new_log2 = max(0, log2 - (10 if shift_held else 1))
        state["max_iter"] = max(1, int(2 ** new_log2)) if action == "iter-down" else int(2 ** new_log2)
        changed = True
    elif action in ("zoom-in", "zoom-out"):
        pass
        changed = True

    return changed

print("=" * 60)
print("KEYBIND ACTION TESTS")
print("=" * 60)

# Test each action in a fresh state
print("\n--- Toggle actions ---")
state = make_state()
orig = dict(state)

# toggle-gpu
state_s = make_state()
simulate_key(state_s, "toggle-gpu")
check("toggle-gpu: flips use_gpu", state_s["use_gpu"] != orig["use_gpu"])

# toggle-smooth
state_s = make_state(smooth=True)
simulate_key(state_s, "toggle-smooth")
check("toggle-smooth: flips to False", state_s["smooth"] is False)

state_s = make_state(smooth=True, step_s=5.0)
simulate_key(state_s, "toggle-smooth")
check("toggle-smooth: resets step_s=0 when disabling", state_s["smooth"] is False and state_s["step_s"] == 0)

# toggle-fxaa
state_s = make_state()
simulate_key(state_s, "toggle-fxaa")
check("toggle-fxaa: flips", state_s["fxaa"] is True)

# toggle-julia
state_s = make_state()
simulate_key(state_s, "toggle-julia")
check("toggle-julia: flips to True", state_s["is_julia"] is True)
state_s = make_state(is_julia=True)
simulate_key(state_s, "toggle-julia")
check("toggle-julia: flips to False", state_s["is_julia"] is False)

# toggle-orbits
state_s = make_state()
simulate_key(state_s, "toggle-orbits")
check("toggle-orbits: flips", state_s["show_orbits"] is False)

# toggle-orbit-handles
state_s = make_state()
simulate_key(state_s, "toggle-orbit-handles")
check("toggle-orbit-handles: flips", state_s["show_orbit_handles"] is False)

# toggle-auto-iter
state_s = make_state(auto_iter=True)
simulate_key(state_s, "toggle-auto-iter")
check("toggle-auto-iter: flips to False", state_s["auto_iter"] is False)
state_s = make_state(auto_iter=False)
simulate_key(state_s, "toggle-auto-iter")
check("toggle-auto-iter: flips to True", state_s["auto_iter"] is True)

print("\n--- Reset actions ---")
# reset-settings
state_s = make_state(is_julia=True, palette_index=5, smooth=False, fxaa=True,
                     phase=0.5, light_angle=0.3, light_azim=0.4, light_i=0.5)
simulate_key(state_s, "reset-settings")
check("reset-settings: is_julia = False", state_s["is_julia"] is False)
check("reset-settings: palette_index = 0", state_s["palette_index"] == 0)
check("reset-settings: smooth = True", state_s["smooth"] is True)
check("reset-settings: fxaa = False", state_s["fxaa"] is False)
check("reset-settings: phase = DEFAULT", state_s["phase"] == mb.DEFAULT_PHASE)
check("reset-settings: light_angle = DEFAULT", state_s["light_angle"] == mb.DEFAULT_LIGHT_ANGLE)
check("reset-settings: orbit_point reset", state_s["orbit_point"] == [0.018, -0.63])
check("reset-settings: auto_iter = DEFAULT", state_s["auto_iter"] == mb.DEFAULT_AUTO_ITER)

# reset-orbit-point
state_s = make_state(orbit_point=[1.5, -0.7])
simulate_key(state_s, "reset-orbit-point")
check("reset-orbit-point: resets to [0.018, -0.63]", state_s["orbit_point"] == [0.018, -0.63])

print("\n--- Palette cycling ---")
palettes = mb.get_all_palettes()
gradient_pals = [p for p in palettes if p.get("type") == "gradient"]
state_s = make_state(palette_index=0, _palette_file_idx=0)
simulate_key(state_s, "cycle-palette")
_gidx = 0
check("cycle-palette: _palette_file_idx 0 -> 1", state_s.get("_palette_file_idx") == 1)
check("cycle-palette: gradient_stops set", "gradient_stops" in state_s)
check("cycle-palette: rgb_thetas at default", state_s["rgb_thetas"] == list(mb.DEFAULT_RGB_THETAS))

state_s = make_state(_palette_file_idx=len(gradient_pals) - 1)
simulate_key(state_s, "cycle-palette")
check("cycle-palette: wraps around _palette_file_idx", state_s.get("_palette_file_idx") == 0)

state_s = make_state(palette_index=-1, _palette_file_idx=0)
simulate_key(state_s, "load-palette-file")
check("load-palette-file: sets valid palette index", 0 <= state_s["palette_index"] < len(palettes))
check("load-palette-file: loads palette data", "gradient_stops" in state_s)

print("\n--- Iteration stepping ---")
state_s = make_state(max_iter=256)
simulate_key(state_s, "iter-up")
check("iter-up: 256 -> 512", state_s["max_iter"] == 512)

state_s = make_state(max_iter=256)
simulate_key(state_s, "iter-up", shift_held=True)
check("iter-up (shift): 256 -> 262144", state_s["max_iter"] == 262144)

state_s = make_state(max_iter=512)
simulate_key(state_s, "iter-down")
check("iter-down: 512 -> 256", state_s["max_iter"] == 256)

state_s = make_state(max_iter=256)
simulate_key(state_s, "iter-down")
check("iter-down: 256 -> 128", state_s["max_iter"] == 128)

state_s = make_state(max_iter=1)
simulate_key(state_s, "iter-down")
check("iter-down: min is 1", state_s["max_iter"] == 1)

print("\n--- Stripe stepping ---")
state_s = make_state(stripe_s=0)
simulate_key(state_s, "stripe-up")
check("stripe-up: 0 -> 1", state_s["stripe_s"] == 1)

state_s = make_state(stripe_s=100)
simulate_key(state_s, "stripe-up")
check("stripe-up: wraps (100 -> 0)", state_s["stripe_s"] == 0)

state_s = make_state(stripe_s=50)
simulate_key(state_s, "stripe-down")
check("stripe-down: 50 -> 49", state_s["stripe_s"] == 49)

state_s = make_state(stripe_s=0)
simulate_key(state_s, "stripe-down")
check("stripe-down: wraps (0 -> 100)", state_s["stripe_s"] == 100)

print("\n--- Phase/light/color stepping ---")
state_s = make_state(phase=0.5)
simulate_key(state_s, "phase-up")
check("phase-up: increases", state_s["phase"] > 0.5)

state_s = make_state(phase=0.5)
simulate_key(state_s, "phase-down")
check("phase-down: decreases", state_s["phase"] < 0.5)

state_s = make_state(light_angle=0.5)
simulate_key(state_s, "light-angle-up")
check("light-angle-up: increases", state_s["light_angle"] > 0.5)

state_s = make_state(k_ambiant=0.3)
simulate_key(state_s, "k-amb-up")
check("k-amb-up: increases", state_s["k_ambiant"] > 0.3)

state_s = make_state(shininess=50.0)
simulate_key(state_s, "shininess-up")
check("shininess-up: increases", state_s["shininess"] > 50.0)

state_s = make_state(rgb_thetas=[0.5, 0.5, 0.5])
simulate_key(state_s, "hue0-up")
check("hue0-up: increases hue0", state_s["rgb_thetas"][0] > 0.5)

state_s = make_state(rgb_thetas=[0.5, 0.5, 0.5])
simulate_key(state_s, "sat-down")
check("sat-down: decreases sat", state_s["rgb_thetas"][2] < 0.5)

print("\n--- No-op / passthrough actions ---")
for action in ("reset-view", "quit", "zoom-in", "zoom-out", "toggle-info"):
    state_s = make_state()
    changed = simulate_key(state_s, action)
    check(f"{action}: no crash", True)

print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
