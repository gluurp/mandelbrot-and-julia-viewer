import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings
from decimal import Decimal, localcontext, getcontext

import pygame

try:
    import gmpy2
except ImportError:
    gmpy2 = None

try:
    import mpmath
except ImportError:
    mpmath = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np
from numba import njit, prange, float64, int64, config

try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    _CUDA_AVAILABLE = False

try:
    from perf_monitor import PerfMonitor, RenderProfiler
    _PERF_MONITOR = PerfMonitor(interval=0.5)
    _RENDER_PROFILER = RenderProfiler()
except Exception:
    _PERF_MONITOR = None
    _RENDER_PROFILER = None

if not os.environ.get("MB_DEBUG_PERF"):
    warnings.filterwarnings("ignore", message=".*Grid size.*GPU under-utilization.*",
                            category=Warning)
    warnings.filterwarnings("ignore", module="numba.cuda.dispatcher")

print(f"Numba threads: {config.NUMBA_NUM_THREADS} | CUDA: {_CUDA_AVAILABLE}")

_dbg_file = None


def _dbg(msg):
    if _dbg_file is not None:
        _dbg_file.write(f"{time.time():.3f} {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\\n")
        _dbg_file.flush()


getcontext().prec = max(getcontext().prec, 80)

DEFAULT_PRECISION_MODE = "auto"
DEFAULT_PRECISION_BITS = 256
DEFAULT_MAX_PRECISION_BITS = 4096
DEFAULT_MAX_AUTO_ITER = 65536
MAX_PRECISION_BITS = DEFAULT_MAX_PRECISION_BITS
DEFAULT_SHOW_UNCERTAINTY = True
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
PRECISION_MODE_CHOICES = ("auto", "mpfr", "float64", "perturbed")


def _normalize_precision_mode(mode):
    normalized = str(mode or DEFAULT_PRECISION_MODE).lower().replace("-", "_")
    return normalized if normalized in PRECISION_MODE_CHOICES else DEFAULT_PRECISION_MODE


def _normalize_precision_bits(bits, max_bits=None):
    try:
        value = int(bits)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_PRECISION_BITS
    cap = MAX_PRECISION_BITS if max_bits is None else int(max_bits)
    cap = max(53, cap)
    return max(53, min(cap, value))


def _as_decimal(value):
    if isinstance(value, Decimal):
        return value
    if gmpy2 is not None and isinstance(value, gmpy2.mpfr):
        return Decimal(str(value))
    if mpmath is not None and isinstance(value, mpmath.mpf):
        return Decimal(str(value))
    return Decimal(str(value))


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _decimal_range(xmin, xmax, ymin, ymax):
    xmin, xmax, ymin, ymax = (_as_decimal(xmin), _as_decimal(xmax),
                              _as_decimal(ymin), _as_decimal(ymax))
    return max(xmax - xmin, ymax - ymin)


def _decimal_log10(value):
    value = _as_decimal(value)
    if value <= 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = max(80, getcontext().prec)
        return value.log10()


def _decimal_mid(low, high):
    return (_as_decimal(low) + _as_decimal(high)) / Decimal(2)


def _decimal_format(value, decimals):
    value = _as_decimal(value)
    if decimals <= 0:
        return format(value, "f")
    with localcontext() as ctx:
        ctx.prec = max(80, getcontext().prec, decimals + 20)
        quantum = Decimal(1).scaleb(-decimals)
        return format(value.quantize(quantum), "f")


def _precision_bits_for_view(xmin, xmax, ymin, ymax):
    view_range = _decimal_range(xmin, xmax, ymin, ymax)
    if view_range <= 0:
        return DEFAULT_PRECISION_BITS
    with localcontext() as ctx:
        ctx.prec = max(80, getcontext().prec)
        magnitude = max(abs(_as_decimal(xmin)), abs(_as_decimal(xmax)),
                        abs(_as_decimal(ymin)), abs(_as_decimal(ymax)), Decimal(1))
        digits = max(16, int(math.ceil(-view_range.log10())) +
                     int(math.ceil(magnitude.log10())) + 12)
    bits = int(math.ceil(digits * math.log(10) / math.log(2)))
    cap = max(128, MAX_PRECISION_BITS)
    return max(128, min(cap, bits)) if bits > 53 else 53


def _select_precision_tier(xmin, xmax, ymin, ymax, mode="auto"):
    mode = _normalize_precision_mode(mode)
    if mode == "mpfr":
        return "mpfr" if gmpy2 is not None else "float64"
    if mode == "float64":
        return "float64"
    if mode == "perturbed":
        return "perturbed"
    view_range = _decimal_range(xmin, xmax, ymin, ymax)
    if view_range < Decimal("1e-14"):
        return "mpfr" if gmpy2 is not None else "float64"
    if view_range < Decimal("1e-10"):
        return "perturbed"
    return "float64"


def _resolve_precision(mode, bits, view_bounds=None):
    """Normalize precision settings and, in auto mode, fit bits to the view."""
    normalized_mode = _normalize_precision_mode(mode)
    normalized_bits = _normalize_precision_bits(bits)
    if normalized_mode == "auto" and view_bounds is not None:
        normalized_bits = min(normalized_bits,
                              _precision_bits_for_view(*view_bounds))
    return normalized_mode, normalized_bits


def _mpfr_grid(low, high, count, precision_bits):
    if gmpy2 is None or count <= 0:
        return None
    if count == 1:
        return [gmpy2.mpfr(str(_as_decimal(low)), precision_bits)]
    with gmpy2.local_context(precision=precision_bits):
        lo = gmpy2.mpfr(str(_as_decimal(low)), precision_bits)
        hi = gmpy2.mpfr(str(_as_decimal(high)), precision_bits)
        step = (hi - lo) / gmpy2.mpfr(count - 1, precision_bits)
        return [lo + step * i for i in range(count)]


def _mpfr_to_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _color_component(value):
    if 0.0 <= value <= 1.0:
        return int(round(value * 255.0))
    return int(round(max(0.0, min(255.0, value))))


def _color_tuple(color):
    return tuple(_color_component(value) for value in color)



MANDATORY_KEYBINDS = {
    "quit": "q",
    "menu": "m",
}

_MOD_MAP = {
    "ctrl":     pygame.KMOD_CTRL,
    "shift":    pygame.KMOD_SHIFT,
    "alt":      pygame.KMOD_ALT,
    "meta":     pygame.KMOD_GUI,
}

DEFAULT_NCYCLE                          = 32
DEFAULT_STRIPE_S                        = 0.0
DEFAULT_STRIPE_SIG                      = 0.9
DEFAULT_STEP_S                          = 0.0
DEFAULT_LIGHT_ANGLE                     = 0.46  # 0.125/(2*pi) normalized
DEFAULT_LIGHT_AZIM                      = 0.5   # 0.5/(pi/2) normalized
DEFAULT_LIGHT_I                         = 0.75
DEFAULT_K_AMBIANT                       = 0.2
DEFAULT_K_DIFFUSE                       = 0.5
DEFAULT_K_SPECULAR                      = 0.5
DEFAULT_SHININESS                       = 20.0
DEFAULT_JULIA_C                         = [0.394, 0.338]
DEFAULT_ORBIT_POINT_M                   = [0.018, -0.63]
DEFAULT_ORBIT_POINT_J                   = [0.153, 0.473]
DEFAULT_ORBIT_MAX_ITER                  = 200
DEFAULT_SET_BLEND                       = 0.0
DEFAULT_C_POINT_SIZE                    = 5
DEFAULT_SPLIT_MODE                      = None
DEFAULT_SPLIT_ORIENT                    = "horizontal"
DEFAULT_JULIA_VIEWPORT                  = (-1.5, 1.5, -1.5, 1.5)  # independent view bounds for Julia pane
DEFAULT_GRID_LABEL_OPACITY              = 1.0
DEFAULT_GRID_OPACITY                    = 0.4
DEFAULT_AUTO_ITER                       = False
DEFAULT_SHOW_POINT_INFO                 = False
DEFAULT_POINT_OUTPUT_FORMAT             = "a+bi"
DEFAULT_CYCLE_OPACITY                   = 0.9
DEFAULT_CYCLE_SIZE                      = 5
DEFAULT_CENTER_OPACITY                   = 0.9
DEFAULT_CENTER_SIZE                      = 7
DEFAULT_HIGHLIGHT_POINTS                = False
DEFAULT_HIGHLIGHT_FLASH_DURATION_MS     = 1000
DEFAULT_HIGHLIGHT_FLASH_FREQUENCY       = 0.02
DEFAULT_POINT_SIZE                      = 5
DEFAULT_ORBITS_OPACITY_M                = 1.0
DEFAULT_ORBITS_OPACITY_J                = 1.0
DEFAULT_LINES_OPACITY_M                 = 1.0
DEFAULT_LINES_OPACITY_J                 = 1.0
NCOL = 2 ** 12
# Classification tuning is configurable via the settings yaml (loaded by
# `_apply_caps_from_settings`). There is deliberately no separate
# `CLASSIFY_MAX_ITER`: classification runs on the current orbit iteration
# budget (`max-orbits`), so the drawn orbit and the classifier agree.
DEFAULT_CLASSIFY_K_MAX      = 60
DEFAULT_CLASSIFY_WINDOWS    = 4
DEFAULT_CLASSIFY_STRICT_REL = 1e-6
DEFAULT_CLASSIFY_NEAR_REL   = 1e-4
CLASSIFY_K_MAX      = DEFAULT_CLASSIFY_K_MAX
CLASSIFY_WINDOWS    = DEFAULT_CLASSIFY_WINDOWS
CLASSIFY_STRICT_REL = DEFAULT_CLASSIFY_STRICT_REL
CLASSIFY_NEAR_REL   = DEFAULT_CLASSIFY_NEAR_REL
DEFAULT_ORBIT_ITER_CAP = 65536
DEFAULT_RENDER_ITER_CAP   = 2 ** 15
ORBIT_ITER_CAP = DEFAULT_ORBIT_ITER_CAP
RENDER_ITER_CAP   = DEFAULT_RENDER_ITER_CAP

PALETTE_FILE    = "palettes.yaml"
SETTINGS_FILE   = "mandelbrot_testing_playground.yaml"
PALETTE_FALLBACK_COLOR = (1.0, 1.0, 1.0)

# Gradient blend modes and the fallback used when no palette is available.
PALETTE_BLEND_MODES    = ("smooth", "linear")
DEFAULT_GRADIENT_BLEND = "smooth"
DEFAULT_GRADIENT_STOPS = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]
_GRADIENT_POSITION_EPS = 1e-9

PALETTE_SECTION_KEYS = [
    "main color",
    "mb set", "ju set",
    "mb lines", "ju lines",
    "mb orbit point", "ju orbit point",
    "mb orbits", "ju orbits",
    "cycles", "center",
    "julia c point", "ju constant",
    "mb orbit drag point", "ju orbit drag point",
    "main menu", "keybinds menu", "points menu", "grid",
]


_all_palettes_cache         = None


def _palette_rewrite_enabled():
    """Allow tests/CI to load palettes without rewriting the tracked file."""
    return os.environ.get("MB_PALETTE_REWRITE", "1").strip().lower() not in ("0", "false", "no")


def _ensure_palettes_loaded():
    global _all_palettes_cache
    if _all_palettes_cache is None:
        palettes = []
        _rewrite = _palette_rewrite_enabled()
        if os.path.exists(PALETTE_FILE):
            try:
                palettes = load_palette_file(PALETTE_FILE, rewrite=_rewrite)
            except Exception as e:
                print(f"[palette] Error loading {PALETTE_FILE}: {e}")
        if not palettes:
            try:
                palettes = load_palette_file(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), PALETTE_FILE),
                    rewrite=False)
            except Exception:
                pass
        if not palettes:
            print(f"[palette] {PALETTE_FILE} not found or empty, no palettes available")
        _all_palettes_cache = palettes


def get_all_palettes():
    _ensure_palettes_loaded()
    return _all_palettes_cache


AUTO_ITER_MAX       = DEFAULT_MAX_AUTO_ITER
AUTO_ITER_MIN       = 32
RENDER_TIMEOUT_MS   = 5000
ZOOM_BASE_ITER      = 64
ANIM_FPS            = 30
ANIM_DURATION       = 10.0
PERSISTENCE_DEBOUNCE_MS = 1000


def _load_settings_checked(filename):
    """Load settings from YAML. Returns ``(settings, ok)``.

    ``ok`` is False only when the file exists but could not be parsed, so
    callers can avoid overwriting a file they could not read.
    """
    settings = {}
    if yaml is None:
        print(f"[settings] PyYAML not available, cannot load {filename}")
        return settings, not os.path.exists(filename)
    try:
        with open(filename) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"'{filename}' not found — will create it as needed.")
        return settings, True
    except Exception as exc:
        print(f"[settings] Error loading {filename}: {exc}")
        return settings, False
    if isinstance(data, dict):
        for key, value in data.items():
            if value is None:
                settings[str(key)] = "None"
            elif isinstance(value, (list, tuple)):
                settings[str(key)] = ",".join(str(v) for v in value)
            else:
                settings[str(key)] = str(value)
    return settings, True


def load_settings(filename):
    """Load settings from YAML."""
    settings, _ = _load_settings_checked(filename)
    return settings


SETTINGS_GROUPS = [
("Persistence",
      lambda k: k == "persistent"),
    ("Viewport",
     lambda k: (k.startswith("view-")
                or k in ("orbit-x", "orbit-y", "orbit-mx", "orbit-my",
                         "orbit-jx", "orbit-jy", "orbit-point",
                         "orbit-point-size",
                         "julia-viewport", "split-mode", "split-orientation",
                         "julia-cx", "julia-cy", "julia-c"))),
    ("Rendering",
     lambda k: k in ("max-iterations", "max-orbits", "auto-iter",
                     "precision-mode", "precision-bits", "show-uncertainty",
                     "max-precision-bits", "max-auto-iterations",
                     "classify-k-max", "classify-windows",
                     "classify-strict-rel", "classify-near-rel",
                     "max-orbits-cap", "max-iterations-cap")),
    ("Lighting",
     lambda k: (k.startswith("light-") or k.startswith("k-")
                or k == "shininess")),
    ("Colors & palette",
     lambda k: (k in ("stripe-s", "stripe-sig", "step-s", "palette-index",
                      "palette-file-idx", "set-blend")
                or k.startswith("point-")
                or k.startswith("c-point")
                or k.startswith("cycle-")
                or k.startswith("center-")
                or k.startswith("orbits-opacity")
                or k.startswith("lines-opacity"))),
    ("Display & grid",
     lambda k: k in ("smooth", "fxaa", "use-gpu",
                     "grid-opacity", "grid-label-opacity")),
    ("Points & visibility",
     lambda k: (k.startswith("highlight-")
                or k in ("point-output-format", "show-keybinds",
                         "show-point-info", "c-point-size"))),
    ("Keybinds", lambda k: k.startswith("keybind.")),
]


def _settings_group_for(key):
    normalized = key.replace("_", "-")
    for index, (title, matcher) in enumerate(SETTINGS_GROUPS):
        try:
            if matcher(normalized):
                return index, title
        except Exception:
            continue
    return len(SETTINGS_GROUPS), "Other"


def _format_setting_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def save_settings(filename, settings):
    temp_filename = filename + ".tmp"
    try:
        save_data = {k: v for k, v in settings.items() if not k.startswith("_")}
        grouped = {}
        for key, value in save_data.items():
            _, title = _settings_group_for(key)
            grouped.setdefault(title, []).append((key, value))
        order = [title for title, _ in SETTINGS_GROUPS] + ["Other"]

        def _sort_key(item):
            key = item[0]
            if key.startswith("keybind."):
                return ("", key[len("keybind."):])
            return (key, "")

        with open(temp_filename, "w") as f:
            first_group = True
            for title in order:
                items = grouped.get(title)
                if not items:
                    continue
                if not first_group:
                    f.write("\n")
                first_group = False
                f.write(f"# {title}\n")
                for key, value in sorted(items, key=_sort_key):
                    f.write(f"{key}: {json.dumps(_format_setting_value(value))}\n")
        os.replace(temp_filename, filename)
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def _validated_viewport(values, default):
    try:
        result = [float(value) for value in values]
        if (len(result) == 4 and all(math.isfinite(value) for value in result) and
                result[1] > result[0] and result[3] > result[2]):
            return result
    except (TypeError, ValueError, OverflowError):
        pass
    return list(default)


def _parse_main_viewport(settings):
    return _validated_viewport(
        [settings.get(key) for key in
         ("view-xmin", "view-xmax", "view-ymin", "view-ymax")],
        [-2.5, 1.0, -1.5, 1.5])


def _parse_julia_viewport(settings):
    """Load julia_viewport from settings, falling back to default."""
    return _validated_viewport(
        str(settings.get("julia-viewport", "")).split(","),
        DEFAULT_JULIA_VIEWPORT)


def _format_viewport(values, default):
    return ",".join(str(value) for value in _validated_viewport(values, default))


_PERSISTENT_GLOBAL_VALUE_GETTERS = {
    "max-precision-bits": lambda: MAX_PRECISION_BITS,
    "max-auto-iterations": lambda: AUTO_ITER_MAX,
}


def _build_persistent_values(state, xmin, xmax, ymin, ymax, keybinds=None):
    values = {}
    for out_key, state_key, default in _PERSISTENT_VALUE_SPECS:
        getter = _PERSISTENT_GLOBAL_VALUE_GETTERS.get(out_key)
        if getter is not None:
            values[out_key] = str(getter())
        elif state_key is None:
            values[out_key] = str(default)
        elif default is _REQUIRED:
            values[out_key] = str(state[state_key])
        else:
            values[out_key] = str(state.get(state_key, default))
    values["julia-cx"] = str(state.get("julia_c", DEFAULT_JULIA_C)[0])
    values["julia-cy"] = str(state.get("julia_c", DEFAULT_JULIA_C)[1])
    values["orbit-mx"] = str(state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[0])
    values["orbit-my"] = str(state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[1])
    values["orbit-jx"] = str(state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[0])
    values["orbit-jy"] = str(state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[1])
    values["view-xmin"] = str(xmin)
    values["view-xmax"] = str(xmax)
    values["view-ymin"] = str(ymin)
    values["view-ymax"] = str(ymax)
    values["julia-viewport"] = _format_viewport(
        state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT),
        DEFAULT_JULIA_VIEWPORT)
    if keybinds is not None:
        for action, kc in keybinds.items():
            if kc is not None:
                values[f"keybind.{action}"] = kc
    return values


_REQUIRED = object()

_PERSISTENT_VALUE_SPECS = [
    ("max-iterations", "max_iter", _REQUIRED),
    ("use-gpu", "use_gpu", _REQUIRED),
    ("light_angle", "light_angle", _REQUIRED),
    ("light_azim", "light_azim", _REQUIRED),
    ("light_i", "light_i", _REQUIRED),
    ("k_ambiant", "k_ambiant", _REQUIRED),
    ("k_diffuse", "k_diffuse", _REQUIRED),
    ("k_specular", "k_specular", _REQUIRED),
    ("shininess", "shininess", _REQUIRED),
    ("stripe_s", "stripe_s", _REQUIRED),
    ("step_s", "step_s", _REQUIRED),
    ("smooth", "smooth", _REQUIRED),
    ("fxaa", "fxaa", False),
    ("set-blend", "set_blend", 0.0),
    ("max-orbits", "orbit_max_iter", _REQUIRED),
    ("orbit-point-size", "orbit_point_size", 3),
    ("c-point-size", "c_point_size", 5),
    ("palette-index", "palette_index", 0),
    ("palette-file-idx", "_palette_file_idx", 0),
    ("split-mode", "split_mode", DEFAULT_SPLIT_MODE),
    ("split-orientation", "split_orientation", DEFAULT_SPLIT_ORIENT),
    ("grid-label-opacity", "grid_label_opacity", DEFAULT_GRID_LABEL_OPACITY),
    ("grid-opacity", "grid_opacity", DEFAULT_GRID_OPACITY),
    ("auto-iter", "auto_iter", DEFAULT_AUTO_ITER),
    ("precision-mode", "precision_mode", DEFAULT_PRECISION_MODE),
    ("precision-bits", "precision_bits", DEFAULT_PRECISION_BITS),
    ("show-uncertainty", "show_uncertainty", DEFAULT_SHOW_UNCERTAINTY),
    ("show-point-info", "show_point_info", DEFAULT_SHOW_POINT_INFO),
    ("point-output-format", "point_output_format", DEFAULT_POINT_OUTPUT_FORMAT),
    ("cycle-opacity", "cycle_opacity", DEFAULT_CYCLE_OPACITY),
    ("cycle-size", "cycle_size", DEFAULT_CYCLE_SIZE),
    ("center-opacity", "center_opacity", DEFAULT_CENTER_OPACITY),
    ("center-size", "center_size", DEFAULT_CENTER_SIZE),
    ("highlight-points", "highlight_points", DEFAULT_HIGHLIGHT_POINTS),
    ("show-keybinds", "show_keybinds", False),
    ("highlight-flash-duration-ms", "highlight_flash_duration_ms", DEFAULT_HIGHLIGHT_FLASH_DURATION_MS),
    ("highlight-flash-frequency", "highlight_flash_frequency", DEFAULT_HIGHLIGHT_FLASH_FREQUENCY),
    ("point-size", "point_size", DEFAULT_POINT_SIZE),
    ("point-size-m", "point_size_m", DEFAULT_POINT_SIZE),
    ("point-size-j", "point_size_j", DEFAULT_POINT_SIZE),
    ("point-size-c", "point_size_c", DEFAULT_POINT_SIZE),
    ("point-opacity-m", "point_opacity_m", 1.0),
    ("point-opacity-j", "point_opacity_j", 1.0),
    ("point-opacity-c", "point_opacity_c", 1.0),
    ("orbits-opacity-m", "orbits_opacity_m", DEFAULT_ORBITS_OPACITY_M),
    ("orbits-opacity-j", "orbits_opacity_j", DEFAULT_ORBITS_OPACITY_J),
    ("lines-opacity-m", "lines_opacity_m", DEFAULT_LINES_OPACITY_M),
    ("lines-opacity-j", "lines_opacity_j", DEFAULT_LINES_OPACITY_J),
    ("max-precision-bits", None, MAX_PRECISION_BITS),
    ("max-auto-iterations", None, AUTO_ITER_MAX),
]




def _persistent_snapshot(values):
    return tuple(sorted(values.items()))


def _save_persistent_state(settings, state, xmin, xmax, ymin, ymax,
                           keybinds, persistent, last_snapshot, last_change,
                           force=False, settings_file=SETTINGS_FILE, now=None):
    values = _build_persistent_values(state, xmin, xmax, ymin, ymax, keybinds)
    snapshot = _persistent_snapshot(values)
    now = time.monotonic() if now is None else now
    if snapshot != last_snapshot:
        last_change = now
    if persistent and (force or
                       (last_change is not None and
                        now - last_change >= PERSISTENCE_DEBOUNCE_MS / 1000.0)):
        settings.update(values)
        try:
            save_settings(settings_file, settings)
        except OSError as exc:
            print(f"[settings] Could not save {settings_file}: {exc}",
                  file=sys.stderr)
            return snapshot, last_change
        return snapshot, None
    return snapshot, last_change


def _normalize_stop_color(values):
    """Normalize an RGB triple from 0-1 or 0-255 into 0-1 floats"""
    if any(value > 1.0 for value in values):
        return tuple(max(0.0, min(1.0, value / 255.0)) for value in values)
    return tuple(max(0.0, min(1.0, value)) for value in values)


def _section_stops_from_raw(raw):
    """Parse a {stops: [[pos, r, g, b], ...]} section into normalized stops"""
    if not isinstance(raw, dict):
        return None
    raw_stops = raw.get("stops")
    if not isinstance(raw_stops, (list, tuple)) or not raw_stops:
        return None
    parsed = []
    for stop in raw_stops:
        if not isinstance(stop, (list, tuple)) or len(stop) < 2:
            continue
        try:
            pos = float(stop[0])
            if pos > 1.0:
                pos /= 255.0
            pos = max(0.0, min(1.0, pos))
            rgb = _normalize_stop_color([float(v) for v in stop[1:4]])
            if len(rgb) < 3:
                continue
            parsed.append((pos, rgb))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0])
    return parsed


def _stops_to_raw(stops):
    """Convert normalized stops back to the example.yaml 0-255 YAML shape."""
    return [[round(pos, 6)] + [int(round(c * 255.0)) for c in rgb]
            for pos, rgb in stops]


def _format_stop_list(stops, indent):
    out = [f"{indent}stops:"]
    for stop in stops:
        out.append(f"{indent}  - [{stop[0]}, {int(stop[1])}, {int(stop[2])}, {int(stop[3])}]")
    return out


def _format_palette_yaml(entries):
    """Serialize palettes in the concise example.yaml layout"""
    lines = ["palettes:"]
    for entry in entries:
        lines.append(f"  - name: {entry['name']}")
        if entry.get("blend") and entry["blend"] != DEFAULT_GRADIENT_BLEND:
            lines.append(f"    blend: {entry['blend']}")
        if "unified color" in entry:
            lines.append("    unified color:")
            lines.extend(_format_stop_list(entry["unified color"]["stops"], "      "))
            continue
        lines.append("    main color:")
        lines.extend(_format_stop_list(entry["main color"]["stops"], "      "))
        for key in PALETTE_SECTION_KEYS:
            if key == "main color" or key not in entry:
                continue
            lines.append(f"    {key}:")
            lines.extend(_format_stop_list(entry[key]["stops"], "      "))
    return "\n".join(lines) + "\n"


def _safe_copy_file(src, dst):
    """Copy ``src`` to ``dst`` without following a symlink at ``dst``."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(dst, flags, 0o644)
    with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
        shutil.copyfileobj(inp, out)


def _rewrite_palette_file(filename, entries):
    """Atomically rewrite the palette file with normalized sections"""
    if yaml is None:
        return
    try:
        if os.path.exists(filename):
            _safe_copy_file(filename, filename + ".bak")
        directory = os.path.dirname(os.path.abspath(filename))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".palette-", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(_format_palette_yaml(entries))
        os.replace(tmp, filename)
    except OSError as exc:
        print(f"[palette] could not rewrite {filename}: {exc}", file=sys.stderr)


def load_palette_file(filename, rewrite=True):
    """Load palettes in the ``example.yaml`` section format

    ``unified color`` collapses all sections; otherwise missing sections fall
    back to ``main color`` in memory and the file is rewritten when it holds
    redundant or duplicate sections

    Returns ``{"name", "gradient_stops", "palette_sections"}`` dicts
    """
    palettes = []
    if not os.path.exists(filename):
        return palettes
    if yaml is None:
        try:
            __import__("yaml")
        except ImportError:
            print(f"[palette] PyYAML not available, cannot load {filename}")
            return palettes
    try:
        with open(filename) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        print(f"[palette] Error loading {filename}: {exc}")
        return palettes
    if not data or "palettes" not in data:
        return palettes

    default_stops = [(0.0, PALETTE_FALLBACK_COLOR)]
    normalized_entries = []
    needs_rewrite = False
    seen_names = set()

    for entry in data["palettes"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        if name in seen_names:
            print(f"[palette] duplicate palette '{name}', using the last definition",
                  file=sys.stderr)
            needs_rewrite = True
        seen_names.add(name)

        raw_blend = entry.get("blend")
        blend = _normalize_gradient_blend(raw_blend)
        if raw_blend is not None:
            if str(raw_blend).strip().lower() != blend:
                needs_rewrite = True
            elif blend == DEFAULT_GRADIENT_BLEND:
                needs_rewrite = True

        unified = _section_stops_from_raw(entry.get("unified color"))
        explicit_sections = set()
        if unified is not None:
            sections = {key: unified for key in PALETTE_SECTION_KEYS}
            explicit_sections = set(PALETTE_SECTION_KEYS)
            canonical = {"name": name,
                         "unified color": {"stops": _stops_to_raw(unified)}}
            if set(entry.keys()) - {"blend"} != {"name", "unified color"}:
                needs_rewrite = True
        else:
            main_stops = _section_stops_from_raw(entry.get("main color"))
            if main_stops is None:
                print(f"[palette] '{name}' has no main color, using the default",
                      file=sys.stderr)
                main_stops = default_stops
                needs_rewrite = True
            sections = {"main color": main_stops}
            canonical = {"name": name, "main color": {"stops": _stops_to_raw(main_stops)}}
            if _section_stops_from_raw(entry.get("main color")) is not None:
                explicit_sections.add("main color")
            for key in PALETTE_SECTION_KEYS:
                if key == "main color":
                    continue
                raw_stops = _section_stops_from_raw(entry.get(key))
                sections[key] = raw_stops if raw_stops is not None else main_stops
                if raw_stops is not None:
                    explicit_sections.add(key)
                    if raw_stops != main_stops:
                        canonical[key] = {"stops": _stops_to_raw(raw_stops)}
                    else:
                        needs_rewrite = True

        if blend != DEFAULT_GRADIENT_BLEND:
            canonical["blend"] = blend

        palettes.append({
            "name": name,
            "gradient_stops": sections["main color"],
            "palette_sections": sections,
            "explicit_sections": explicit_sections,
            "blend": blend,
        })
        normalized_entries.append(canonical)

    if rewrite and needs_rewrite:
        _rewrite_palette_file(filename, normalized_entries)
    return palettes


def get_palette_section(palette, key, fallback_to_main=True):
    """Return a section's stops from a palette, falling back to main color"""
    sections = (palette or {}).get("palette_sections") or {}
    if key in sections:
        return sections[key]
    if fallback_to_main:
        return sections.get("main color") or (palette or {}).get("gradient_stops")
    return None


def _active_palette(state):
    palettes = get_all_palettes()
    if not palettes:
        return None
    idx = state.get("palette_index", 0)
    if 0 <= idx < len(palettes):
        return palettes[idx]
    return palettes[0]


def _palette_section_explicit(state, key):
    """Return section stops only when the palette defined them explicitly"""
    pal = _active_palette(state)
    if pal is None:
        return None
    if key not in pal.get("explicit_sections", ()): 
        return None
    return (pal.get("palette_sections") or {}).get(key)


def _palette_section_stops(state, key):
    pal = _active_palette(state)
    if pal is None:
        return None
    return get_palette_section(pal, key, fallback_to_main=True)


def _active_gradient_blend(state):
    """Blend mode for the active palette (falls back to the default)"""
    return state.get("gradient_blend", DEFAULT_GRADIENT_BLEND)


_SECTION_COLORTABLE_CACHE: dict = {}
_SECTION_COLORTABLE_CACHE_MAX = 32


def _palette_section_colortable_or(state, key, fallback):
    """Use an explicit palette section, else keep the caller's colortable.

    Built colortables are cached by (section, stops, blend) because this runs
    on the per-frame orbit draw path.
    """
    stops = _palette_section_explicit(state, key)
    if not stops:
        return fallback
    blend = _active_gradient_blend(state)
    cache_key = (key, tuple((pos, tuple(rgb)) for pos, rgb in stops), blend)
    cached = _SECTION_COLORTABLE_CACHE.get(cache_key)
    if cached is None:
        cached = _make_gradient_colortable(stops, blend)
        if len(_SECTION_COLORTABLE_CACHE) > _SECTION_COLORTABLE_CACHE_MAX:
            _SECTION_COLORTABLE_CACHE.clear()
        _SECTION_COLORTABLE_CACHE[cache_key] = cached
    return cached


def _palette_section_color(state, key, pos=0.5):
    """Sample a section color as an (r, g, b) 0-255 tuple"""
    stops = _palette_section_stops(state, key)
    if not stops:
        return _color_tuple(PALETTE_FALLBACK_COLOR)
    r, g, b = _gradient_sample(stops, pos, _active_gradient_blend(state))
    return (max(0, min(255, int(round(r * 255.0)))),
            max(0, min(255, int(round(g * 255.0)))),
            max(0, min(255, int(round(b * 255.0)))))


def get_custom_palettes():
    """Load palettes from PALETTE_FILE (cached). Returns list of palette dicts"""
    _ensure_palettes_loaded()
    return _all_palettes_cache


def get_palette_names():
    """Return list of palette names from the loaded file"""
    return [p["name"] for p in get_all_palettes()]


def get_palette_by_name(name):
    """Look up a palette by name from the loaded file"""
    for p in get_all_palettes():
        if p["name"] == name:
            return p
    return None


def _apply_palette_by_name(state, name):
    """Apply a named palette to state. Returns True on success"""
    pal = get_palette_by_name(name)
    if pal is None:
        return False
    state["gradient_stops"] = list(pal["gradient_stops"])
    state["gradient_blend"] = pal.get("blend", DEFAULT_GRADIENT_BLEND)
    return True


def _apply_palette_by_index(state, index):
    """Apply a palette by index from the loaded list. Returns True on success"""
    palettes = get_all_palettes()
    if not (0 <= index < len(palettes)):
        return False
    state["gradient_stops"] = list(palettes[index]["gradient_stops"])
    state["gradient_blend"] = palettes[index].get("blend", DEFAULT_GRADIENT_BLEND)
    return True


def get_keybind(settings, name, default_key=None):
    """Read a keybind from settings; missing or blank means unbound"""
    val = settings.get(f"keybind.{name}")
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default_key
    return val.strip().lower()


def load_keybinds(settings, settings_file=None):
    """Load YAML keybinds and guarantee the q/m safety bindings"""
    keybinds = {}
    for key, value in settings.items():
        if not key.startswith("keybind."):
            continue
        action = key[len("keybind."):]
        if action:
            keybinds[action] = get_keybind(settings, action)

    changed = False
    for action, default_key in MANDATORY_KEYBINDS.items():
        if not keybinds.get(action):
            keybinds[action] = default_key
            settings[f"keybind.{action}"] = default_key
            changed = True

    if changed and settings_file:
        # Merge into the on-disk file instead of replacing it with the caller's
        # (possibly partial or empty) settings dict. Skip the write when the
        # file exists but could not be parsed, so an unreadable settings file is
        # never replaced with just the mandatory keybinds.
        merged, load_ok = _load_settings_checked(settings_file)
        if load_ok:
            for action in keybinds:
                merged[f"keybind.{action}"] = keybinds[action]
            try:
                save_settings(settings_file, merged)
            except OSError as exc:
                print(f"[keybind] could not persist {settings_file}: {exc}", file=sys.stderr)
    return keybinds


def get_persistent_setting(settings, key, cast=str, default=None, prompt=None):
    if key in settings:
        try:
            return cast(settings[key])
        except ValueError:
            print(f"Warning: '{key}' = '{settings[key]}' is invalid, using default.", file=sys.stderr)
    if default is not None:
        settings[key] = str(default)
        return default
    if prompt is not None:
        prompt_text = prompt
        if default is not None:
            prompt_text += f" [default: {default}]"
        prompt_text += ": "
        try:
            raw = input(prompt_text).strip()
            value = default if raw == "" and default is not None else cast(raw)
        except EOFError:
            value = default
        settings[key] = str(value)
        return value
    return None


def get_runtime_input(key, cast=float, prompt=None, default=None):
    prompt_text = (prompt or f"Enter {key}") + ": "
    raw = input(prompt_text).strip()
    return cast(raw) if raw else default


def _apply_caps_from_settings(settings):
    """Load the YAML-configurable maxima into the module globals.

    `max-precision-bits` and `max-auto-iterations` are the only hard ceilings on
    zoom depth; both can be raised arbitrarily in the settings file.
    """
    global MAX_PRECISION_BITS, AUTO_ITER_MAX
    global CLASSIFY_K_MAX, CLASSIFY_WINDOWS, CLASSIFY_STRICT_REL, CLASSIFY_NEAR_REL
    global ORBIT_ITER_CAP, RENDER_ITER_CAP
    bits_cap = get_persistent_setting(settings, "max-precision-bits", cast=int,
                                      default=DEFAULT_MAX_PRECISION_BITS)
    MAX_PRECISION_BITS = max(53, int(bits_cap or DEFAULT_MAX_PRECISION_BITS))
    iter_cap = get_persistent_setting(settings, "max-auto-iterations", cast=int,
                                      default=DEFAULT_MAX_AUTO_ITER)
    AUTO_ITER_MAX = max(AUTO_ITER_MIN, int(iter_cap or DEFAULT_MAX_AUTO_ITER))

    k_cap = get_persistent_setting(settings, "classify-k-max", cast=int,
                                   default=DEFAULT_CLASSIFY_K_MAX)
    CLASSIFY_K_MAX = max(1, int(k_cap or DEFAULT_CLASSIFY_K_MAX))
    w_cap = get_persistent_setting(settings, "classify-windows", cast=int,
                                   default=DEFAULT_CLASSIFY_WINDOWS)
    CLASSIFY_WINDOWS = max(1, int(w_cap or DEFAULT_CLASSIFY_WINDOWS))
    strict = get_persistent_setting(settings, "classify-strict-rel", cast=float,
                                    default=DEFAULT_CLASSIFY_STRICT_REL)
    CLASSIFY_STRICT_REL = float(strict) if strict else DEFAULT_CLASSIFY_STRICT_REL
    near = get_persistent_setting(settings, "classify-near-rel", cast=float,
                                  default=DEFAULT_CLASSIFY_NEAR_REL)
    CLASSIFY_NEAR_REL = float(near) if near else DEFAULT_CLASSIFY_NEAR_REL

    orbit_cap = get_persistent_setting(settings, "max-orbits-cap", cast=int,
                                       default=DEFAULT_ORBIT_ITER_CAP)
    ORBIT_ITER_CAP = max(1, int(orbit_cap or DEFAULT_ORBIT_ITER_CAP))
    render_cap = get_persistent_setting(settings, "max-iterations-cap", cast=int,
                                        default=DEFAULT_RENDER_ITER_CAP)
    RENDER_ITER_CAP = max(1, int(render_cap or DEFAULT_RENDER_ITER_CAP))


@njit
def iterate_single(cx, cy, max_iter):
    zx, zy = 0.0, 0.0

    orbit_x = np.empty(max_iter)
    orbit_y = np.empty(max_iter)

    for i in range(max_iter):
        zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy

        orbit_x[i] = zx
        orbit_y[i] = zy

        if zx * zx + zy * zy > 4:
            return orbit_x[:i + 1], orbit_y[:i + 1], True
    return orbit_x, orbit_y, False


# --- Gradient interpolation -------------------------------------------------
#
# Gradient stops are (position, (r, g, b)) pairs with position and channels in
# [0, 1]. The blend style is a parameter so it can evolve without touching the
# call sites:
#   "linear" - piecewise-linear, one line per channel.
#   "smooth" - monotone cubic Hermite (PCHIP / Fritsch-Carlson): passes through
#              every stop, C1 continuous, and provably never overshoots.
# Both entry points share the helpers below, so a new mode only needs to be
# added in one place.


def _normalize_gradient_blend(blend):
    """Coerce a blend name to a supported mode, defaulting to smooth."""
    if blend is None:
        return DEFAULT_GRADIENT_BLEND
    text = str(blend).strip().lower()
    return text if text in PALETTE_BLEND_MODES else DEFAULT_GRADIENT_BLEND


def _sanitize_gradient_stops(stops):
    """Return sorted, deduped, clamped stops with >= 2 strictly increasing positions."""
    cleaned = []
    for stop in stops or ():
        try:
            position = float(stop[0])
            channels = [float(channel) for channel in stop[1][:3]]
        except (TypeError, ValueError, IndexError):
            continue
        if len(channels) < 3:
            continue
        position = max(0.0, min(1.0, position))
        color = tuple(max(0.0, min(1.0, channel)) for channel in channels)
        cleaned.append((position, color))
    if not cleaned:
        return list(DEFAULT_GRADIENT_STOPS)
    cleaned.sort(key=lambda item: item[0])
    deduped = []
    for position, color in cleaned:
        if deduped and abs(deduped[-1][0] - position) <= _GRADIENT_POSITION_EPS:
            deduped[-1] = (position, color)  # keep the last definition
        else:
            deduped.append((position, color))
    if len(deduped) == 1:
        _, color = deduped[0]
        deduped = [(0.0, color), (1.0, color)]
    return deduped


def _gradient_endpoint_tangent(h0, h1, d0, d1):
    """One-sided Fritsch-Carlson tangent for a spline endpoint."""
    tangent = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if tangent * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(tangent) > 3.0 * abs(d0):
        return 3.0 * d0
    return tangent


def _gradient_tangents(xs, ys):
    """Monotone cubic tangents (Fritsch-Carlson) for a single channel."""
    count = len(xs)
    widths = np.diff(xs)
    secants = np.diff(ys) / widths
    if count == 2:
        return np.array([secants[0], secants[0]], dtype=np.float64)
    tangents = np.zeros(count, dtype=np.float64)
    for i in range(1, count - 1):
        previous, current = secants[i - 1], secants[i]
        if previous * current <= 0.0:
            tangents[i] = 0.0
            continue
        w1 = 2.0 * widths[i] + widths[i - 1]
        w2 = widths[i] + 2.0 * widths[i - 1]
        tangents[i] = (w1 + w2) / (w1 / previous + w2 / current)
    tangents[0] = _gradient_endpoint_tangent(widths[0], widths[1], secants[0], secants[1])
    tangents[-1] = _gradient_endpoint_tangent(widths[-1], widths[-2], secants[-1], secants[-2])
    return tangents


def _evaluate_cubic_hermite(x, xs, ys, tangents):
    """Evaluate a cubic Hermite spline over x (values ys, slopes tangents)."""
    widths = np.diff(xs)
    segment = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    t = np.clip((x - xs[segment]) / widths[segment], 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (h00 * ys[segment] + h10 * widths[segment] * tangents[segment] +
            h01 * ys[segment + 1] + h11 * widths[segment] * tangents[segment + 1])


def _interpolate_gradient(x, xs, colors, blend):
    """Interpolate an (n, 3) color matrix at positions ``x`` for a blend mode.

    Single dispatch point for blend styles: add a new mode here and in
    ``PALETTE_BLEND_MODES`` and every caller picks it up.
    """
    if _normalize_gradient_blend(blend) == "smooth" and len(xs) > 2:
        columns = [
            _evaluate_cubic_hermite(
                x, xs, colors[:, channel], _gradient_tangents(xs, colors[:, channel]))
            for channel in range(3)
        ]
        return np.clip(np.column_stack(columns), 0.0, 1.0)
    return np.clip(
        np.column_stack([np.interp(x, xs, colors[:, channel]) for channel in range(3)]),
        0.0, 1.0)


def _gradient_sample(stops, position, blend=DEFAULT_GRADIENT_BLEND):
    """Sample one (r, g, b) triplet in [0, 1] at ``position``."""
    stops = _sanitize_gradient_stops(stops)
    xs = np.array([stop[0] for stop in stops], dtype=np.float64)
    colors = np.array([stop[1] for stop in stops], dtype=np.float64)
    position = max(0.0, min(1.0, float(position)))
    sample = _interpolate_gradient(np.array([position]), xs, colors, blend)[0]
    return tuple(float(value) for value in sample)


def _make_gradient_colortable(stops, blend=DEFAULT_GRADIENT_BLEND):
    """Build a (NCOL, 3) float32 colortable from gradient stops.

    Args:
        stops: list of (position, (r, g, b)) with position and channels in [0, 1].
        blend: "smooth" (monotone cubic, default) or "linear".

    Returns:
        np.ndarray of shape (NCOL, 3) with float32 values in [0, 1].
    """
    stops = _sanitize_gradient_stops(stops)
    xs = np.array([stop[0] for stop in stops], dtype=np.float64)
    colors = np.array([stop[1] for stop in stops], dtype=np.float64)
    x = np.linspace(0.0, 1.0, NCOL)
    return _interpolate_gradient(x, xs, colors, blend).astype(np.float32)


@njit
def overlay(x, y, gamma):
    if (2 * y) < 1:
        out = 2 * x * y
    else:
        out = 1 - 2 * (1 - x) * (1 - y)
    return out * gamma + x * (1 - gamma)


@njit
def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@njit
def blinn_phong(normal, light):
    mag = abs(normal)
    if mag == 0.0:
        return 0.0
    normal = normal / mag
    ldiff = (normal.real * math.cos(light[0]) * math.cos(light[1]) +
             normal.imag * math.sin(light[0]) * math.cos(light[1]) +
             math.sin(light[1]))
    ldiff = ldiff / (1 + math.sin(light[1]))
    phi_half = (math.pi / 2 + light[1]) / 2
    lspec = (normal.real * math.cos(light[0]) * math.sin(phi_half) +
             normal.imag * math.sin(light[0]) * math.sin(phi_half) +
             math.cos(phi_half))
    lspec = lspec / (1 + math.cos(phi_half))
    lspec = lspec ** light[6]
    bright = light[3] + light[4] * ldiff + light[5] * lspec
    bright = bright * light[2] + (1 - light[2]) / 2
    return bright


@njit
def color_pixel(niter, stripe_a, step_s, dem, normal, colortable, ncycle, light, smooth=True):
    ncol = colortable.shape[0] - 1
    if smooth:
        cniter = math.sqrt(niter) % ncycle / ncycle
        col_i = min(ncol, round(cniter * ncol))
    else:
        floor_n = float(int(math.sqrt(niter)))
        col_i = round(floor_n / ncycle * ncol) % ncol
        cniter = (niter % ncycle) / ncycle
    bright = blinn_phong(normal, light)
    dem = 1e-10 if 1e-10 > dem else dem
    dem = -math.log(dem) / 12
    dem = 1.0 / (1.0 + math.exp(-10 * ((2 * dem - 1) / 2)))

    nshader = 0
    shader = 0.0
    if stripe_a > 0:
        nshader += 1
        shader += stripe_a
    if step_s > 0:
        n_steps = 1.0 if 1.0 > step_s else step_s
        quantized = math.floor(cniter * n_steps) / n_steps
        col_i = round(quantized * ncol)
        x = (cniter - quantized) * n_steps
        light_step = 6 * (1 - x ** 5 - (1 - x) ** 100) / 10
        x2 = (cniter - quantized) * n_steps * 8
        x2f = x2 - math.floor(x2)
        light_step2 = 6 * (1 - x2f ** 5 - (1 - x2f) ** 30) / 10
        light_step = overlay(light_step2, light_step, 1)
        nshader += 1
        shader += light_step
    if nshader > 0:
        bright = overlay(bright, shader / nshader, 1) * (1 - dem) + dem * bright

    r = overlay(colortable[col_i, 0], bright, 1)
    g = overlay(colortable[col_i, 1], bright, 1)
    b = overlay(colortable[col_i, 2], bright, 1)

    return (_clamp01(r), _clamp01(g), _clamp01(b))


@njit
def _smooth_iter_scalar(cx, cy, maxiter, stripe_s, stripe_sig, use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    esc_radius_2 = 10.0**10
    if use_julia:
        zr, zi = cx, cy
        c_re, c_im = julia_c_re, julia_c_im
    else:
        zr, zi = 0.0, 0.0
        c_re, c_im = cx, cy
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a = 0.0
    dzr, dzi = 1.0, 0.0
    for n in range(maxiter):
        new_dzr = dzr * 2 * zr - dzi * 2 * zi + 1
        new_dzi = dzi * 2 * zr + dzr * 2 * zi
        dzr, dzi = new_dzr, new_dzi

        new_zr = zr * zr - zi * zi + c_re
        new_zi = 2 * zr * zi + c_im
        zr, zi = new_zr, new_zi

        if stripe:
            stripe_t = (math.sin(stripe_s * math.atan2(zi, zr)) + 1) / 2

        if zr * zr + zi * zi > esc_radius_2:
            modz = math.sqrt(zr * zr + zi * zi)
            log_ratio = 2 * math.log(modz) / math.log(esc_radius_2)
            smooth_i = 1 - math.log(log_ratio) / math.log(2)
            if stripe:
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig - 1)) +
                            stripe_t * smooth_i * (1 - stripe_sig))
                stripe_a = stripe_a / (1 - stripe_sig ** n *
                                       (1 + smooth_i * (stripe_sig - 1)))
            normal_re = zr * dzr + zi * dzi
            normal_im = zi * dzr - zr * dzi
            ndem = math.sqrt(dzr * dzr + dzi * dzi)
            if ndem > 0:
                normal_re = normal_re / ndem
                normal_im = normal_im / ndem
            dem = modz * math.log(modz) / ndem / 2
            return (n + smooth_i, stripe_a, dem, complex(normal_re, normal_im))
        if stripe:
            stripe_a = stripe_a * stripe_sig + stripe_t * (1 - stripe_sig)
    return (0.0, 0.0, 0.0, 0j)


@njit
def _smooth_iter_numba(cx, cy, maxiter, stripe_s, stripe_sig, use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    """Smooth iteration count for Mandelbrot/Julia sets (internal numba signature)."""
    return _smooth_iter_scalar(cx, cy, maxiter, stripe_s, stripe_sig, use_julia, julia_c_re, julia_c_im)


@njit(parallel=True)
def compute_set_cpu(creal, cim, maxiter, colortable, ncycle,
                    stripe_s, stripe_sig, step_s, diag, light, smooth=True,
                    use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3), dtype=np.float64)
    for x in prange(xpixels):
        cx = creal[x]
        for y in range(ypixels):
            cy = cim[y]
            niter, stripe_a, dem, normal = _smooth_iter_numba(
                cx, cy, maxiter, stripe_s, stripe_sig,
                use_julia, julia_c_re, julia_c_im)
            if niter > 0:
                r, g, b = color_pixel(niter, stripe_a, step_s, dem / diag,
                                      normal, colortable, ncycle, light, smooth)
                mat[y, x, 0] = r
                mat[y, x, 1] = g
                mat[y, x, 2] = b
    return mat


_REF_ORBIT_CACHE: dict = {}
_REF_ORBIT_CACHE_MAX = 8


def needs_perturbation(xmin, xmax, ymin, ymax):
    """Check if perturbation theory should be used based on view range."""
    return _decimal_range(xmin, xmax, ymin, ymax) < Decimal("1e-10")


@njit
def compute_reference_orbit(cx, cy, max_iter):
    """Compute float64 reference orbit for perturbation theory.

    Returns np.ndarray((max_iter+1, 2)) with (real, imag) columns.
    """
    zr, zi = 0.0, 0.0
    orbit = np.empty((max_iter + 1, 2), dtype=np.float64)
    orbit[0, 0] = zr
    orbit[0, 1] = zi
    for n in range(max_iter):
        zr, zi = zr * zr - zi * zi + cx, 2.0 * zr * zi + cy
        orbit[n + 1, 0] = zr
        orbit[n + 1, 1] = zi
        if zr * zr + zi * zi > 1e10:
            return orbit[:n + 1]
    return orbit[:max_iter + 1]


@njit
def compute_reference_orbit_julia(z0_re, z0_im, max_iter,
                                  julia_c_re, julia_c_im):
    """Compute a Julia reference orbit from its initial point."""
    zr, zi = z0_re, z0_im
    orbit = np.empty((max_iter + 1, 2), dtype=np.float64)
    orbit[0, 0] = zr
    orbit[0, 1] = zi
    for n in range(max_iter):
        zr, zi = zr * zr - zi * zi + julia_c_re, 2.0 * zr * zi + julia_c_im
        orbit[n + 1, 0] = zr
        orbit[n + 1, 1] = zi
        if zr * zr + zi * zi > 1e10:
            return orbit[:n + 1]
    return orbit[:max_iter + 1]


def _get_ref_orbit(cx, cy, max_iter):
    """Get reference orbit from cache or compute it."""
    key = (round(cx, 10), round(cy, 10), max_iter)
    if key in _REF_ORBIT_CACHE:
        return _REF_ORBIT_CACHE[key]
    orbit = compute_reference_orbit(cx, cy, max_iter)
    _REF_ORBIT_CACHE[key] = orbit
    if len(_REF_ORBIT_CACHE) > _REF_ORBIT_CACHE_MAX:
        _REF_ORBIT_CACHE.clear()
    return orbit


def _get_ref_orbit_julia(z0_re, z0_im, max_iter, julia_c_re, julia_c_im):
    """Get a cached Julia reference orbit."""
    key = (round(z0_re, 10), round(z0_im, 10), max_iter,
           round(julia_c_re, 10), round(julia_c_im, 10))
    if key in _REF_ORBIT_CACHE:
        return _REF_ORBIT_CACHE[key]
    orbit = compute_reference_orbit_julia(z0_re, z0_im, max_iter,
                                           julia_c_re, julia_c_im)
    _REF_ORBIT_CACHE[key] = orbit
    if len(_REF_ORBIT_CACHE) > _REF_ORBIT_CACHE_MAX:
        _REF_ORBIT_CACHE.clear()
    return orbit


@njit(parallel=True)
def compute_set_perturbed(creal_arr, cim_arr, ref_orbit, max_iter,
                           colortable, ncycle, stripe_s, stripe_sig,
                           step_s, diag, light, smooth, glitch_threshold=1e-4,
                           use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    """Compute the set using perturbation theory.

    Reference orbit is computed in float64. Per-pixel deltas use float32.
    Output is float64 to match CPU lighting precision.
    Glitch detection: if |delta| > glitch_threshold, recompute pixel directly.
    """
    xpixels = len(creal_arr)
    ypixels = len(cim_arr)
    mat = np.zeros((ypixels, xpixels, 3), dtype=np.float64)
    ref_len = ref_orbit.shape[0]

    if use_julia:
        ref_z0_x = ref_orbit[0, 0]
        ref_z0_y = ref_orbit[0, 1]
        ref_cx = julia_c_re
        ref_cy = julia_c_im
    else:
        ref_z0_x = 0.0
        ref_z0_y = 0.0
        if ref_len > 1:
            ref_cx = ref_orbit[1, 0]
            ref_cy = ref_orbit[1, 1]
        else:
            ref_cx = 0.0
            ref_cy = 0.0

    use_stripe = (stripe_s > 0.0) and (stripe_sig > 0.0)

    for x in prange(xpixels):
        for y in range(ypixels):
            creal = creal_arr[x]
            cim = cim_arr[y]

            if use_julia:
                dr = creal - ref_z0_x
                di = cim - ref_z0_y
            else:
                dr = 0.0
                di = 0.0
            zr = ref_z0_x + dr
            zi = ref_z0_y + di
            dzr, dzi = 1.0, 0.0
            stripe_a = 0.0
            glitch = False
            escaped = False
            n = 0

            for n in range(min(max_iter, ref_len - 1)):
                zr_ref = ref_orbit[n, 0]
                zi_ref = ref_orbit[n, 1]
                delta_c_re = 0.0 if use_julia else creal - ref_cx
                delta_c_im = 0.0 if use_julia else cim - ref_cy
                new_dr = (2.0 * zr_ref * dr - 2.0 * zi_ref * di
                          + dr * dr - di * di + delta_c_re)
                new_di = (2.0 * zr_ref * di + 2.0 * zi_ref * dr
                          + 2.0 * dr * di + delta_c_im)
                dr, di = new_dr, new_di
                zr = zr_ref + dr
                zi = zi_ref + di

                new_dzr = dzr * 2.0 * zr - dzi * 2.0 * zi + 1.0
                new_dzi = dzi * 2.0 * zr + dzr * 2.0 * zi
                dzr, dzi = new_dzr, new_dzi

                if use_stripe:
                    stripe_t = (math.sin(stripe_s * math.atan2(zi, zr)) + 1.0) / 2.0
                    stripe_a = stripe_a * stripe_sig + stripe_t * (1.0 - stripe_sig)

                if dr * dr + di * di > glitch_threshold * glitch_threshold:
                    glitch = True
                    break

                if zr * zr + zi * zi > 1e10:
                    escaped = True
                    break

            if glitch:
                if use_julia:
                    zr, zi = creal, cim
                else:
                    zr, zi = 0.0, 0.0
                dzr, dzi = 1.0, 0.0
                stripe_a = 0.0
                escaped = False
                for n in range(max_iter):
                    new_dzr = dzr * 2.0 * zr - dzi * 2.0 * zi + 1.0
                    new_dzi = dzi * 2.0 * zr + dzr * 2.0 * zi
                    dzr, dzi = new_dzr, new_dzi

                    if use_julia:
                        new_zr = zr * zr - zi * zi + julia_c_re
                        new_zi = 2.0 * zr * zi + julia_c_im
                    else:
                        new_zr = zr * zr - zi * zi + creal
                        new_zi = 2.0 * zr * zi + cim
                    zr, zi = new_zr, new_zi

                    if use_stripe:
                        stripe_t = (math.sin(stripe_s * math.atan2(zi, zr)) + 1.0) / 2.0
                        stripe_a = stripe_a * stripe_sig + stripe_t * (1.0 - stripe_sig)

                    if zr * zr + zi * zi > 1e10:
                        escaped = True
                        break

            modz = math.sqrt(zr * zr + zi * zi)
            if escaped and modz > 2.0:
                smooth_i = 0.0
                log_ratio = 2.0 * math.log(modz) / math.log(1e10)
                if log_ratio > 0.0:
                    smooth_i = 1.0 - math.log(log_ratio) / math.log(2.0)
                if use_stripe:
                    stripe_a = (stripe_a * (1.0 + smooth_i * (stripe_sig - 1.0)) +
                                stripe_t * smooth_i * (1.0 - stripe_sig))
                    stripe_a = stripe_a / (1.0 - stripe_sig ** n *
                                           (1.0 + smooth_i * (stripe_sig - 1.0)))
                niter = float(n) + smooth_i
            else:
                niter = 0.0

            if niter > 0.0:
                ndem = math.sqrt(dzr * dzr + dzi * dzi)
                if ndem > 0.0:
                    normal_re = (zr * dzr + zi * dzi) / ndem
                    normal_im = (zi * dzr - zr * dzi) / ndem
                    dem = modz * math.log(modz) / ndem / 2.0
                else:
                    normal_re = 0.0
                    normal_im = 0.0
                    dem = 0.0

                r, g, b = color_pixel(niter, stripe_a, step_s, dem / diag,
                                      complex(normal_re, normal_im),
                                      colortable, ncycle, light, smooth)
                mat[y, x, 0] = r
                mat[y, x, 1] = g
                mat[y, x, 2] = b
    return mat


def compute_set_mpfr(x_values, y_values, max_iter, colortable, ncycle,
                     stripe_s, stripe_sig, step_s, diag, light, smooth=True,
                     use_julia=False, julia_c_re=0.0, julia_c_im=0.0,
                     precision_bits=DEFAULT_PRECISION_BITS,
                     show_uncertainty=DEFAULT_SHOW_UNCERTAINTY):
    if gmpy2 is None:
        warnings.warn("gmpy2 unavailable; high-precision render fell back to float64",
                      RuntimeWarning)
        x_array = np.asarray([_as_float(value) for value in x_values], dtype=np.float64)
        y_array = np.asarray([_as_float(value) for value in y_values], dtype=np.float64)
        mat = compute_set_cpu(x_array, y_array, max_iter, colortable, ncycle,
                              stripe_s, stripe_sig, step_s,
                              _mpfr_to_float(diag), light, smooth,
                              use_julia, _mpfr_to_float(julia_c_re),
                              _mpfr_to_float(julia_c_im))
        return mat, np.zeros((len(y_array), len(x_array)), dtype=bool)

    precision_bits = max(53, min(MAX_PRECISION_BITS, int(precision_bits)))
    mat = np.zeros((len(y_values), len(x_values), 3), dtype=np.float64)
    uncertain = np.zeros((len(y_values), len(x_values)), dtype=bool)
    esc_radius_2 = gmpy2.mpfr("1e10", precision_bits)
    julia_c_re = gmpy2.mpfr(str(_as_decimal(julia_c_re)), precision_bits)
    julia_c_im = gmpy2.mpfr(str(_as_decimal(julia_c_im)), precision_bits)
    stripe_s_mp = gmpy2.mpfr(str(_as_decimal(stripe_s)), precision_bits)
    stripe_sig_mp = gmpy2.mpfr(str(_as_decimal(stripe_sig)), precision_bits)
    use_stripe = stripe_s > 0.0 and stripe_sig > 0.0
    diag_float = _mpfr_to_float(diag, 1.0)

    with gmpy2.local_context(precision=precision_bits):
        for y, cim in enumerate(y_values):
            cim = gmpy2.mpfr(str(_as_decimal(cim)), precision_bits)
            for x, creal in enumerate(x_values):
                creal = gmpy2.mpfr(str(_as_decimal(creal)), precision_bits)
                if use_julia:
                    zr, zi = creal, cim
                    c_re, c_im = julia_c_re, julia_c_im
                else:
                    zr, zi = gmpy2.mpfr(0, precision_bits), gmpy2.mpfr(0, precision_bits)
                    c_re, c_im = creal, cim

                dzr, dzi = gmpy2.mpfr(1, precision_bits), gmpy2.mpfr(0, precision_bits)
                stripe_a = gmpy2.mpfr(0, precision_bits)
                stripe_t = gmpy2.mpfr(0, precision_bits)
                escaped = False
                n = 0
                for n in range(max_iter):
                    new_dzr = dzr * 2 * zr - dzi * 2 * zi + 1
                    new_dzi = dzi * 2 * zr + dzr * 2 * zi
                    dzr, dzi = new_dzr, new_dzi
                    new_zr = zr * zr - zi * zi + c_re
                    new_zi = 2 * zr * zi + c_im
                    zr, zi = new_zr, new_zi
                    if use_stripe:
                        stripe_t = (gmpy2.sin(stripe_s_mp * gmpy2.atan2(zi, zr)) + 1) / 2
                        stripe_a = stripe_a * stripe_sig_mp + stripe_t * (1 - stripe_sig_mp)
                    if zr * zr + zi * zi > esc_radius_2:
                        escaped = True
                        break

                if not escaped:
                    uncertain[y, x] = True
                    continue

                modz = gmpy2.sqrt(zr * zr + zi * zi)
                log_ratio = 2 * gmpy2.log(modz) / gmpy2.log(esc_radius_2)
                smooth_i = gmpy2.mpfr(1) - gmpy2.log(log_ratio) / gmpy2.log(2)
                niter = float(n + smooth_i)
                if use_stripe:
                    stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig - 1)) +
                                stripe_t * smooth_i * (1 - stripe_sig))
                    denominator = 1 - stripe_sig ** n * (1 + smooth_i * (stripe_sig - 1))
                    if denominator != 0:
                        stripe_a /= denominator
                ndem = gmpy2.sqrt(dzr * dzr + dzi * dzi)
                if ndem > 0:
                    normal_re = (zr * dzr + zi * dzi) / ndem
                    normal_im = (zi * dzr - zr * dzi) / ndem
                    dem = modz * gmpy2.log(modz) / ndem / 2
                else:
                    normal_re = normal_im = dem = gmpy2.mpfr(0, precision_bits)
                r, g, b = color_pixel(
                    niter, _mpfr_to_float(stripe_a), step_s,
                    _mpfr_to_float(dem) / diag_float,
                    complex(_mpfr_to_float(normal_re), _mpfr_to_float(normal_im)),
                    colortable, ncycle, light, smooth)
                mat[y, x, 0] = r
                mat[y, x, 1] = g
                mat[y, x, 2] = b

    if show_uncertainty and uncertain.any():
        mat[uncertain] = (1.0, 0.0, 1.0)
    return mat, uncertain


if cuda is not None:
    @cuda.jit(fastmath=True)
    def compute_set_gpu(mat, width, height, total_w, total_h, x_start, y_start, x_min, x_max, y_min, y_max, maxiter, colortable,
                        ncycle, stripe_s, stripe_sig, step_s, diag, light, smooth, use_julia, julia_c_re, julia_c_im):
        index = cuda.grid(1)
        x = index % width
        y = index // width
        if y < height:
            creal   = x_min + (x_start + x) / (1 if 1 > total_w - 1 else total_w - 1) * (x_max - x_min)
            cim     = y_min + (y_start + y) / (1 if 1 > total_h - 1 else total_h - 1) * (y_max - y_min)
            ncol = colortable.shape[0] - 1
            # Inline smooth iteration
            esc_radius_2 = 10.0**10
            if use_julia:
                zr, zi = creal, cim
            else:
                zr, zi = 0.0, 0.0
            dzr, dzi = 1.0, 0.0
            niter = 0.0
            stripe_a = 0.0
            dem = 0.0
            normal_re, normal_im = 0.0, 0.0
            use_stripe = (stripe_s > 0) and (stripe_sig > 0)

            for n in range(maxiter):
                new_dzr = dzr * 2 * zr - dzi * 2 * zi + 1
                new_dzi = dzi * 2 * zr + dzr * 2 * zi
                dzr, dzi = new_dzr, new_dzi

                if use_julia:
                    new_zr = zr * zr - zi * zi + julia_c_re
                    new_zi = 2 * zr * zi + julia_c_im
                else:
                    new_zr = zr * zr - zi * zi + creal
                    new_zi = 2 * zr * zi + cim
                zr, zi = new_zr, new_zi

                if use_stripe:
                    stripe_t = (math.sin(stripe_s * math.atan2(zi, zr)) + 1) / 2

                if zr * zr + zi * zi > esc_radius_2:
                    modz = math.sqrt(zr * zr + zi * zi)
                    log_ratio = 2 * math.log(modz) / math.log(esc_radius_2)
                    smooth_i = 1 - math.log(log_ratio) / math.log(2)
                    niter = float(n + smooth_i)
                    dem = modz * math.log(modz) / math.sqrt(dzr * dzr + dzi * dzi) / 2
                    normal_re = zr * dzr + zi * dzi
                    normal_im = zi * dzr - zr * dzi
                    ndem = math.sqrt(dzr * dzr + dzi * dzi)
                    if ndem > 0:
                        normal_re = normal_re / ndem
                        normal_im = normal_im / ndem
                    if use_stripe:
                        stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig - 1)) +
                                    stripe_t * smooth_i * (1 - stripe_sig))
                        denom = 1 - stripe_sig ** n * (1 + smooth_i * (stripe_sig - 1))
                        stripe_a = stripe_a / denom
                    break

                if use_stripe:
                    stripe_a = stripe_a * stripe_sig + stripe_t * (1 - stripe_sig)

            if niter > 0:
                if smooth:
                    cniter = math.sqrt(niter) % ncycle / ncycle
                    col_i = int(round(cniter * ncol))
                    if col_i > ncol:
                        col_i = ncol
                else:
                    floor_n = float(int(math.sqrt(niter)))
                    col_i = int(round(floor_n / ncycle * ncol)) % ncol
                    cniter = (niter % ncycle) / ncycle
                nshader_aa = 0.0

                # Inline blinn_phong
                mag = math.sqrt(normal_re * normal_re + normal_im * normal_im)
                if mag > 0:
                    nr = normal_re / mag
                    ni = normal_im / mag
                    ldiff = (nr * math.cos(light[0]) * math.cos(light[1]) +
                             ni * math.sin(light[0]) * math.cos(light[1]) +
                             math.sin(light[1]))
                    ldiff = ldiff / (1 + math.sin(light[1]))
                    phi_half = (math.pi / 2 + light[1]) / 2
                    lspec = (nr * math.cos(light[0]) * math.sin(phi_half) +
                             ni * math.sin(light[0]) * math.sin(phi_half) +
                             math.cos(phi_half))
                    lspec = lspec / (1 + math.cos(phi_half))
                    lspec = math.pow(lspec, light[6])
                    bright = light[3] + light[4] * ldiff + light[5] * lspec
                    bright = bright * light[2] + (1 - light[2]) / 2
                else:
                    bright = 0.0
                dem = dem / diag
                dem = 1e-10 if 1e-10 > dem else dem
                dem = -math.log(dem) / 12
                dem = 1.0 / (1.0 + math.exp(-10 * ((2 * dem - 1) / 2)))
                nshader = 0
                shader = 0.0
                if stripe_a > 0:
                    nshader += 1
                    shader += stripe_a
                if step_s > 0:
                    n_steps = 1.0 if 1.0 > step_s else step_s
                    quantized = math.floor(cniter * n_steps) / n_steps
                    col_i = int(round(quantized * ncol))
                    if col_i > ncol:
                        col_i = ncol
                    x2 = (cniter - quantized) * n_steps
                    light_step = 6 * (1 - math.pow(x2, 5) - math.pow(1 - x2, 100)) / 10
                    x8 = (cniter - quantized) * n_steps * 8
                    x8f = x8 - math.floor(x8)
                    light_step2 = 6 * (1 - math.pow(x8f, 5) - math.pow(1 - x8f, 30)) / 10
                    if 2 * light_step < 1:
                        ls = 2 * light_step * light_step2
                    else:
                        ls = 1 - 2 * (1 - light_step) * (1 - light_step2)
                    nshader += 1
                    shader += ls
                if nshader > 0:
                    sh = shader / nshader
                    if 2 * sh < 1:
                        bright = (2 * bright * sh) * (1 - dem) + dem * bright
                    else:
                        bright = (1 - 2 * (1 - bright) * (1 - sh)) * (1 - dem) + dem * bright
                # Inline overlay for color lookup (gamma=1)
                cr = colortable[col_i, 0]
                cg = colortable[col_i, 1]
                cb = colortable[col_i, 2]
                if 2 * bright < 1:
                    mat[y, x, 0] = 2 * cr * bright
                else:
                    mat[y, x, 0] = 1 - 2 * (1 - cr) * (1 - bright)
                if 2 * bright < 1:
                    mat[y, x, 1] = 2 * cg * bright
                else:
                    mat[y, x, 1] = 1 - 2 * (1 - cg) * (1 - bright)
                if 2 * bright < 1:
                    mat[y, x, 2] = 2 * cb * bright
                else:
                    mat[y, x, 2] = 1 - 2 * (1 - cb) * (1 - bright)
else:
    compute_set_gpu = None



def build_render_params(state, maxiter=None, use_julia=False, view_bounds=None):
    """Build compute_image params for a single set mode.

    Pass use_julia=True to get Julia params (renders z^2 + c_J where c_J = state['julia_c']).
    Pass use_julia=False for Mandelbrot params (each pixel is c).
    The blending of both sets is handled by render_to_surface.
    """
    gradient_stops = state.get("gradient_stops") or DEFAULT_GRADIENT_STOPS
    colortable = _make_gradient_colortable(gradient_stops, _active_gradient_blend(state))
    light = np.array([
        state["light_angle"] * 2 * math.pi,
        state["light_azim"] * math.pi / 2,
        state["light_i"],
        state["k_ambiant"],
        state["k_diffuse"],
        state["k_specular"],
        state["shininess"],
    ], dtype=np.float64)
    julia_c = state.get("julia_c", DEFAULT_JULIA_C)
    precision_mode, precision_bits = _resolve_precision(
        state.get("precision_mode", DEFAULT_PRECISION_MODE),
        state.get("precision_bits", DEFAULT_PRECISION_BITS),
        view_bounds)
    return {
        "colortable": colortable,
        "ncy": math.sqrt(maxiter) if maxiter else math.sqrt(DEFAULT_NCYCLE),
        "stripe_s": state["stripe_s"],
        "stripe_sig": state["stripe_sig"],
        "step_s": state["step_s"],
        "light": light,
        "use_gpu": state["use_gpu"] and _CUDA_AVAILABLE,
        "smooth": state.get("smooth", True),
        "fxaa": state.get("fxaa", False),
        "use_julia": use_julia,
        "julia_c_re": julia_c[0],
        "julia_c_im": julia_c[1],
        "set_blend": state.get("set_blend", 0.0),
        "precision_mode": precision_mode,
        "precision_bits": precision_bits,
        "show_uncertainty": bool(state.get("show_uncertainty", DEFAULT_SHOW_UNCERTAINTY)),
        "view_bounds": tuple(_as_decimal(value) for value in view_bounds) if view_bounds else None,
    }


def _post_process(mat, apply_aa=False):
    mat = mat[::-1, :, :]
    if apply_aa:
        mat = apply_fxaa(mat)
    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)
    mat = np.clip(mat, 0.0, 1.0)
    rgb = (mat * 255.0).astype(np.uint8)
    rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    return rgb_t


@njit(parallel=True)
def apply_fxaa(mat):
    h, w = mat.shape[0], mat.shape[1]
    out = np.empty_like(mat)
    for y in prange(1, h - 1):
        for x in range(1, w - 1):
            l = mat[y, x-1, 0] * 0.299 + mat[y, x-1, 1] * 0.587 + mat[y, x-1, 2] * 0.114
            r = mat[y, x+1, 0] * 0.299 + mat[y, x+1, 1] * 0.587 + mat[y, x+1, 2] * 0.114
            t = mat[y-1, x, 0] * 0.299 + mat[y-1, x, 1] * 0.587 + mat[y-1, x, 2] * 0.114
            b = mat[y+1, x, 0] * 0.299 + mat[y+1, x, 1] * 0.587 + mat[y+1, x, 2] * 0.114
            c = mat[y, x, 0] * 0.299 + mat[y, x, 1] * 0.587 + mat[y, x, 2] * 0.114
            mn = c
            mx = c
            for v in [l, r, t, b]:
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            rng = mx - mn
            if rng < 0.0625:
                out[y, x, 0] = mat[y, x, 0]
                out[y, x, 1] = mat[y, x, 1]
                out[y, x, 2] = mat[y, x, 2]
            else:
                gx = l + r - 2 * c
                gy = t + b - 2 * c
                if abs(gx) > abs(gy):
                    n = 1 if l > r else -1
                    px = max(0, min(w - 1, x + n))
                    n_val = 0.5
                    out[y, x, 0] = mat[y, x, 0] * (1 - n_val) + mat[y, px, 0] * n_val
                    out[y, x, 1] = mat[y, x, 1] * (1 - n_val) + mat[y, px, 1] * n_val
                    out[y, x, 2] = mat[y, x, 2] * (1 - n_val) + mat[y, px, 2] * n_val
                else:
                    n = 1 if t > b else -1
                    py = max(0, min(h - 1, y + n))
                    n_val = 0.5
                    out[y, x, 0] = mat[y, x, 0] * (1 - n_val) + mat[py, x, 0] * n_val
                    out[y, x, 1] = mat[y, x, 1] * (1 - n_val) + mat[py, x, 1] * n_val
                    out[y, x, 2] = mat[y, x, 2] * (1 - n_val) + mat[py, x, 2] * n_val
    for x in range(w):
        out[0, x, 0] = mat[0, x, 0]
        out[0, x, 1] = mat[0, x, 1]
        out[0, x, 2] = mat[0, x, 2]
        out[h-1, x, 0] = mat[h-1, x, 0]
        out[h-1, x, 1] = mat[h-1, x, 1]
        out[h-1, x, 2] = mat[h-1, x, 2]
    for y in range(1, h - 1):
        out[y, 0, 0] = mat[y, 0, 0]
        out[y, 0, 1] = mat[y, 0, 1]
        out[y, 0, 2] = mat[y, 0, 2]
        out[y, w-1, 0] = mat[y, w-1, 0]
        out[y, w-1, 1] = mat[y, w-1, 1]
        out[y, w-1, 2] = mat[y, w-1, 2]
    return out


def _blit_surface_clamped(screen, surface, dx, dy, screen_w, screen_h):
    sw, sh = surface.get_size()
    sx = max(0, -dx)
    sy = max(0, -dy)
    dx_adj = dx + sx
    dy_adj = dy + sy
    blit_w = min(sw - sx, screen_w - dx_adj)
    blit_h = min(sh - sy, screen_h - dy_adj)
    if blit_w > 0 and blit_h > 0:
        screen.blit(surface, (dx_adj, dy_adj), area=(sx, sy, blit_w, blit_h))


@njit
def compute_orbit_numba(sx, sy, max_iter, use_julia, julia_c_re, julia_c_im):
    """Numba-accelerated orbit computation. Returns (n_points, points_array)."""
    points = np.empty((max_iter + 1, 2))
    points[0, 0] = sx
    points[0, 1] = sy
    if use_julia:
        zx = sx
        zy = sy
        const_x = julia_c_re
        const_y = julia_c_im
    else:
        zx = 0.0
        zy = 0.0
        const_x = sx
        const_y = sy
    for i in range(max_iter):
        new_zx = zx * zx - zy * zy + const_x
        new_zy = 2.0 * zx * zy + const_y
        zx = new_zx
        zy = new_zy
        points[i + 1, 0] = zx
        points[i + 1, 1] = zy
        if zx * zx + zy * zy > 4.0:
            return i + 2, points[:i + 2]
    return max_iter + 1, points


def compute_orbit(sx, sy, max_iter, use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    """Compute the orbit of point (sx, sy).

    Mandelbrot mode: z0=0, c=(sx,sy) → z_{n+1} = z_n^2 + c
    Julia mode: z0=(sx,sy), c=(julia_c_re, julia_c_im) → z_{n+1} = z_n^2 + c_J
    Returns array of (x, y) points."""
    n, points = compute_orbit_numba(sx, sy, max_iter, use_julia, julia_c_re, julia_c_im)
    return points[:n]


def classify_orbit(points, max_iter):
    """Classify an orbit using an auto-thresholded, validated cycle search.

    Python-side enhancement over math.txt's K_max/A_s scheme: the periodicity
    threshold is derived automatically from the orbit's own motion scale
    (CLASSIFY_STRICT_REL / CLASSIFY_NEAR_REL) instead of a fixed absolute
    distance, and a candidate is validated across CLASSIFY_WINDOWS
    consecutive returns so a coincidental near-return is never reported as a
    cycle. The math.txt formulas themselves are unchanged.

    Returns dict with:
      - status: 'escaped', 'superattracting', 'attracting_fixed',
                'repelling_fixed', 'attracting_cycle', 'repelling_cycle',
                'neutral_siegel_candidate', 'near_periodic',
                'bounded_nonperiodic_unknown'
      - proven: True only for a validated cycle (only these are drawn)
      - cycle_pts / cycle_len / center_pt / multiplier / log_mult
      - candidate_len / residual: best unproven candidate (near_periodic)
      - confidence: 0-100 confidence in the reported status
      - escaped: bool
      - evidence: str describing the detection method
    """
    empty = {
        "status": "bounded_nonperiodic_unknown",
        "proven": False,
        "cycle_pts": [],
        "cycle_len": 0,
        "candidate_len": 0,
        "residual": None,
        "center_pt": (0.0, 0.0),
        "mean_pt": (0.0, 0.0),
        "multiplier": None,
        "log_mult": None,
        "confidence": 0,
        "escaped": False,
        "evidence": "empty orbit",
    }
    if len(points) == 0:
        return empty

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return {**empty, "evidence": "orbit too short"}

    last = pts[-1]
    if last[0] * last[0] + last[1] * last[1] > 4.0:
        return {**empty, "status": "escaped", "confidence": 100,
                "escaped": True, "evidence": "orbit escaped",
                "mean_pt": (float(np.mean(pts[:, 0])),
                            float(np.mean(pts[:, 1])))}

    # Motion scale from the whole orbit (not just the settled tail) so a fully
    # converged cluster of cycle points still normalizes correctly.
    steps = np.abs(np.diff(pts, axis=0))
    if len(steps):
        scale = float(np.sqrt(np.max(steps[:, 0] ** 2 + steps[:, 1] ** 2)))
    else:
        scale = 0.0

    def _multiplier(cycle_arr):
        mult = complex(1.0, 0.0)
        log_mult = 0.0
        for pt in cycle_arr:
            term = 2.0 * complex(float(pt[0]), float(pt[1]))
            mult *= term
            mag = abs(term)
            log_mult += math.log(mag) if mag > 0.0 else -100.0
        return mult, log_mult

    def _result(status, proven, cycle_arr, candidate_len, residual, evidence,
                confidence):
        if proven:
            mult, log_mult = _multiplier(cycle_arr)
            center = (float(np.mean(cycle_arr[:, 0])),
                      float(np.mean(cycle_arr[:, 1])))
            mean_pt = center
            cycle_list = cycle_arr.tolist()
            cycle_len = len(cycle_arr)
        else:
            mult, log_mult = None, None
            center = (0.0, 0.0)
            # math.txt mean(C_ML): average of the candidate window (the last
            # `candidate_len` orbit points), else the whole orbit mean.
            if candidate_len and candidate_len <= n:
                tail = pts[n - candidate_len:]
                mean_pt = (float(np.mean(tail[:, 0])),
                           float(np.mean(tail[:, 1])))
            else:
                mean_pt = (float(np.mean(pts[:, 0])),
                           float(np.mean(pts[:, 1])))
            cycle_list = []
            cycle_len = 0
        return {
            "status": status,
            "proven": bool(proven),
            "cycle_pts": cycle_list,
            "cycle_len": cycle_len,
            "candidate_len": int(candidate_len),
            "residual": None if residual is None else float(residual),
            "center_pt": center,
            "mean_pt": mean_pt,
            "multiplier": mult,
            "log_mult": log_mult,
            "confidence": int(confidence),
            "escaped": False,
            "evidence": evidence,
        }

    if scale <= 1e-300:
        # Constant orbit: the itinerary has landed exactly on a fixed point.
        # Classify by that point's multiplier (it need not be attracting).
        z = complex(float(pts[-1][0]), float(pts[-1][1]))
        abs_mult = abs(2.0 * z)
        if abs_mult < 1e-10:
            status = "superattracting"
        elif abs(abs_mult - 1.0) <= 1e-6:
            status = "neutral_siegel_candidate"
        elif abs_mult < 1.0:
            status = "attracting_fixed"
        else:
            status = "repelling_fixed"
        return _result(status, True, pts[-1:].copy(), 1, 0.0,
                       f"constant orbit (fixed point), |lambda|={abs_mult:.6g}",
                       100)

    max_k = min(CLASSIFY_K_MAX, n - CLASSIFY_WINDOWS)
    strict_dist = CLASSIFY_STRICT_REL * scale
    near_dist = CLASSIFY_NEAR_REL * scale

    def _window_residuals(K):
        rmax = 0.0
        residuals = []
        for j in range(CLASSIFY_WINDOWS):
            i = n - 1 - j
            d = pts[i] - pts[i - K]
            r = math.hypot(float(d[0]), float(d[1]))
            residuals.append(r)
            if r > rmax:
                rmax = r
        return rmax, residuals

    best = None          # (rmax, K, residuals) over all candidates
    for K in range(1, max_k + 1):
        rmax, residuals = _window_residuals(K)
        if best is None or rmax < best[0]:
            best = (rmax, K, residuals)

    if best is not None and best[0] <= strict_dist:
        rmax, K, _ = best
        # Minimal period: collapse to the smallest divisor that also validates.
        for d in range(1, K):
            if K % d:
                continue
            d_rmax, _ = _window_residuals(d)
            if d_rmax <= strict_dist:
                K, rmax = d, d_rmax
                break
        cycle_arr = pts[n - K:].copy()
        mult_probe, _ = _multiplier(cycle_arr)
        abs_mult = abs(mult_probe)
        if abs(abs_mult - 1.0) <= 1e-6:
            status = "neutral_siegel_candidate"
        elif abs_mult < 1.0:
            status = "attracting_fixed" if K == 1 else "attracting_cycle"
        else:
            status = "repelling_fixed" if K == 1 else "repelling_cycle"
        if K == 1 and abs_mult < 1e-10:
            status = "superattracting"
        ratio = rmax / scale if scale > 0.0 else 0.0
        confidence = max(50, int(100 * (1.0 - min(1.0, ratio / CLASSIFY_STRICT_REL))))
        return _result(
            status, True, cycle_arr, K, rmax,
            f"cycle proven K={K}, residual={rmax:.3e}, scale={scale:.3e}, "
            f"|lambda|={abs_mult:.6g}",
            confidence)

    # Near-periodic: the smallest K whose return distance is small and forms a
    # sharp dip versus its neighbouring periods. A spiral approach need not be
    # monotone, so only the dip (not the direction) is required.
    near_best = None
    for K in range(1, max_k + 1):
        rmax, residuals = _window_residuals(K)
        if rmax <= 0.0 or rmax > near_dist:
            continue
        neighbours = []
        if K - 1 >= 1:
            neighbours.append(_window_residuals(K - 1)[0])
        if K + 1 <= max_k:
            neighbours.append(_window_residuals(K + 1)[0])
        if neighbours and rmax > 0.5 * min(neighbours):
            continue
        near_best = (rmax, K, residuals)
        break

    if near_best is not None:
        rmax, K, _ = near_best
        return _result("near_periodic", False, None, K, rmax,
                       f"near-periodic candidate K={K}, residual={rmax:.3e}, "
                       f"scale={scale:.3e}", 40)

    if best is not None:
        rmax, K, _ = best
        evidence = (f"no validated cycle; best K={K}, residual={rmax:.3e} "
                    f"({rmax / scale:.2e} of motion scale)")
        return _result("bounded_nonperiodic_unknown", False, None, K, rmax,
                       evidence, 20)
    return _result("bounded_nonperiodic_unknown", False, None, 0, None,
                   "no cycle candidates in window", 20)


def _classify_cached(sx, sy, use_julia, c_re, c_im, max_iter):
    """Classify an orbit, reusing cached orbit points and classification."""
    key = (sx, sy, max_iter, use_julia, c_re, c_im)
    cached = _CLASSIFY_CACHE.get(key)
    if cached is not None:
        return cached
    points = _ORBIT_CACHE.get(key)
    if points is None:
        points = compute_orbit(sx, sy, max_iter, use_julia, c_re, c_im)
        # Long classification orbits are large; the classification result is
        # cached separately, so only keep short orbits in the shared cache.
        if max_iter <= 2048:
            if len(_ORBIT_CACHE) > 256:
                _ORBIT_CACHE.clear()
            _ORBIT_CACHE[key] = points
    cached = classify_orbit(points, max_iter)
    if len(_CLASSIFY_CACHE) > _CLASSIFY_CACHE_MAX:
        _CLASSIFY_CACHE.clear()
    _CLASSIFY_CACHE[key] = cached
    return cached


def _mandelbrot_to_screen_all(points, xmin, xmax, ymin, ymax, width, height):
    """Convert all Mandelbrot coordinates to screen coordinates (including off-screen)."""
    if len(points) == 0:
        return []
    pts = np.asarray(points, dtype=np.float64)
    dx = width / (xmax - xmin) if (xmax - xmin) != 0 else 1.0
    dy = height / (ymax - ymin) if (ymax - ymin) != 0 else 1.0
    sx = (pts[:, 0] - xmin) * dx
    sy = (ymax - pts[:, 1]) * dy
    return np.column_stack((sx, sy))



def _liang_barsky_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Clip a line segment to a rectangle. Returns None if fully outside."""
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u1, u2 = 0.0, 1.0
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
        if u1 > u2:
            return None
    nx0 = x0 + u1 * dx
    ny0 = y0 + u1 * dy
    nx1 = x0 + u2 * dx
    ny1 = y0 + u2 * dy
    return (nx0, ny0, nx1, ny1)


def _nice_num(x, round=True):
    """Return a 'nice' number approximately equal to x."""
    exp = math.floor(math.log10(x))
    f = x / (10 ** exp)
    if round:
        if f < 1.5: nf = 1
        elif f < 3: nf = 2
        elif f < 7: nf = 5
        else: nf = 10
    else:
        if f <= 1: nf = 1
        elif f <= 2: nf = 2
        elif f <= 5: nf = 5
        else: nf = 10
    return nf * (10 ** exp)


def _compute_grid_spacing(xmin, xmax, ymin, ymax, width, height):
    """Compute zoom-adaptive grid spacing using nice intervals (1/2/5 multiples)."""
    rx = xmax - xmin
    ry = ymax - ymin
    if rx <= 0 or ry <= 0:
        return 1.0
    pixels_per_unit_x = width / rx
    pixels_per_unit_y = height / ry
    desired_x = 80.0 / pixels_per_unit_x
    desired_y = 80.0 / pixels_per_unit_y
    sx = _nice_num(desired_x)
    sy = _nice_num(desired_y)
    spacing = min(sx, sy)
    return spacing


def _grid_label_text(value, spacing):
    """Format a grid value, keeping it short at deep zoom."""
    if value == 0 or abs(value) < max(1e-12, abs(spacing) * 1e-6):
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e5:
        return f"{value:.1e}"
    if spacing > 0:
        decimals = max(0, min(8, int(math.ceil(-math.log10(spacing)))))
    else:
        decimals = 3
    text = f"{value:.{decimals}f}"
    return "0" if text in ("-0", "-0.0") else text


def _zoom_percent(xmin, xmax, ymin, ymax, width, height):
    """Zoom level as a percentage relative to the default 4.0-wide view."""
    view_range = max(xmax - xmin, ymax - ymin)
    if view_range <= 0:
        return 0.0
    default_range = 4.0
    if view_range >= default_range:
        return 0.0
    zoom_level = math.log2(default_range / view_range)
    max_zoom_level = math.log2(default_range / 1e-10)
    return min(100.0, (zoom_level / max_zoom_level) * 100.0)


def _run_clipboard_tool(cmd, text):
    """Run a clipboard tool, feeding it text on stdin. True on exit code 0."""
    import subprocess
    try:
        proc = subprocess.run(cmd, input=text.encode("utf-8"),
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=2.0)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _clipboard_backends():
    """Ordered clipboard backends for this session, as (name, runner) pairs.

    Wayland prefers ``wl-copy``; X11 prefers SDL's ``pygame.scrap``; both fall
    back to ``xclip``/``xsel``. Every backend degrades gracefully when absent.
    """
    backends = []
    wl_copy = shutil.which("wl-copy")
    scrap = getattr(pygame, "scrap", None)

    if wl_copy and os.environ.get("WAYLAND_DISPLAY"):
        backends.append(("wl-copy",
                         lambda text: _run_clipboard_tool([wl_copy], text)))

    if scrap is not None:
        def _scrap(text):
            if not pygame.display.get_init():
                return False
            try:
                if not scrap.get_init():
                    scrap.init()
                scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
                return True
            except Exception:
                return False
        backends.append(("pygame.scrap", _scrap))

    for tool, args in (("xclip", ["-selection", "clipboard"]),
                       ("xsel", ["--clipboard", "--input"])):
        path = shutil.which(tool)
        if path:
            backends.append(
                (tool, lambda text, p=path, a=args: _run_clipboard_tool([p] + a, text)))

    return backends


def _set_clipboard(text):
    """Copy ``text`` to the system clipboard.

    Tries each available backend in order and returns True as soon as one
    reports success, so the caller can surface a visible "copy failed" instead
    of failing silently.
    """
    for _name, runner in _clipboard_backends():
        try:
            if runner(text):
                return True
        except Exception:
            continue
    print(f"[clipboard] {text}", file=sys.stderr)
    return False


def _commit_coord_edit(state, coord_input, text):
    """Parse and commit a coordinate edit to state.

    ``coord_input`` is one of "M", "J", "C".  ``text`` is the raw user input.
    Returns True if a valid coordinate pair was parsed and committed.
    """
    if coord_input is None or not text:
        return False
    text = text.strip().replace("i", "")
    parts = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', text)
    if len(parts) < 2:
        return False
    try:
        re_val = float(parts[0])
        im_val = float(parts[1])
    except ValueError:
        return False
    if coord_input == "M":
        state["orbit_point_m"] = [re_val, im_val]
    elif coord_input == "J":
        state["orbit_point_j"] = [re_val, im_val]
    elif coord_input == "C":
        state["julia_c"] = [re_val, im_val]
    return True


def _color_opacity(color, opacity):
    """Scale an RGB 0-255 color by opacity, clamping to valid range."""
    return (max(0, min(255, int(color[0] * opacity))),
            max(0, min(255, int(color[1] * opacity))),
            max(0, min(255, int(color[2] * opacity))))


class PointInfoOverlay:
    """Floating bottom-left overlay showing orbit classification and point data.

    Toggled via the main menu (show_point_info). Uses a single format toggle
    (a+bi <-> (x,y)) and copy buttons; coordinate editing is handled by
    MenuOverlay.
    """

    def __init__(self):
        self.active = False
        self._cache = None
        self._cache_key = None
        self.button_rects = {}
        self._menu_cache = None
        self.tab = 0
        self.collapsed = False
        self.scroll = 0
        self.pos = None            # [x, y] top-left once dragged
        self.user_moved = False
        self.dragging = False
        self.drag_offset = (0, 0)
        self.panel_rect = None     # absolute panel rect from the last draw
        self.content_rect = None   # scrollable content viewport
        self._copy_feedback = None  # (ok: bool, ticks_ms) shown briefly

    @staticmethod
    def _fmt(point, fmt):
        re, im = point[0], point[1]
        if fmt == "(x, y)":
            return f"({re:.15g}, {im:.15g})"
        sign = "+" if im >= 0 else "-"
        return f"{re:.15g} {sign} {abs(im):.15g}i"

    def _gather_data(self, state, xmin, xmax, ymin, ymax, width, height, pane_bounds=None):
        """Compute the classification and viewport data for current point selection."""
        orbit_pt_m = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
        orbit_pt_j = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
        julia_c = state.get("julia_c", DEFAULT_JULIA_C)
        classify_iter = state.get("orbit_max_iter", DEFAULT_ORBIT_MAX_ITER)

        cls_m = _classify_cached(orbit_pt_m[0], orbit_pt_m[1], False,
                                 julia_c[0], julia_c[1], classify_iter)
        cls_j = _classify_cached(orbit_pt_j[0], orbit_pt_j[1], True,
                                 julia_c[0], julia_c[1], classify_iter)

        if pane_bounds is not None and len(pane_bounds) > 1:
            mb_pane, ju_pane = _resolve_mb_ju_panes(pane_bounds)
            mb_px = _complex_to_screen_pane(orbit_pt_m[0], orbit_pt_m[1], pane=mb_pane,
                                            width=width, height=height)
            ju_px = _complex_to_screen_pane(orbit_pt_j[0], orbit_pt_j[1], pane=ju_pane,
                                            width=width, height=height)
            c_px = _complex_to_screen_pane(julia_c[0], julia_c[1], pane=mb_pane,
                                           width=width, height=height)
        else:
            mb_px = _complex_to_screen_pane(orbit_pt_m[0], orbit_pt_m[1],
                                          width=width, height=height,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            ju_px = _complex_to_screen_pane(orbit_pt_j[0], orbit_pt_j[1],
                                            width=width, height=height,
                                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            c_px = _complex_to_screen_pane(julia_c[0], julia_c[1],
                                          width=width, height=height,
                                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

        fmt = state.get("point_output_format", DEFAULT_POINT_OUTPUT_FORMAT)
        view_range = max(xmax - xmin, ymax - ymin)
        zoom_pct = _zoom_percent(xmin, xmax, ymin, ymax, width, height)
        pixel_scale = view_range / max(1, width)

        center_m = _center_point_for(orbit_pt_m[0], orbit_pt_m[1], cls_m)
        center_j = _center_point_for(julia_c[0], julia_c[1], cls_j)

        return {
            "fmt": fmt,
            "cls_m": cls_m, "cls_j": cls_j,
            "orbit_pt_m": orbit_pt_m, "orbit_pt_j": orbit_pt_j, "julia_c": julia_c,
            "center_m": center_m, "center_j": center_j,
            "mb_px": mb_px, "ju_px": ju_px, "c_px": c_px,
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "zoom_pct": zoom_pct, "pixel_scale": pixel_scale,
        }

    def handle_click(self, pos, state):
        """Handle tab switches, copy, and panel dragging."""
        if not self.active:
            return False, False
        x, y = pos
        if self.panel_rect is not None and self.panel_rect.collidepoint(x, y):
            for i in range(3):
                r = self.button_rects.get(f"tab-{i}")
                if r and pygame.Rect(r).collidepoint(x, y):
                    if self.tab == i:
                        self.collapsed = not self.collapsed
                    else:
                        self.tab = i
                        self.collapsed = False
                        self.scroll = 0
                    return True, False
            r = self.button_rects.get("format")
            if r and pygame.Rect(r).collidepoint(x, y):
                current = state.get("point_output_format", DEFAULT_POINT_OUTPUT_FORMAT)
                state["point_output_format"] = ("(x, y)" if current == "a+bi"
                                                else "a+bi")
                self._menu_cache = None
                return True, False
            r = self.button_rects.get("copy-tab")
            if r and pygame.Rect(r).collidepoint(x, y):
                self._copy_coord(("m", "j", "c")[self.tab], state)
                return True, False
            self.dragging = True
            self.drag_offset = (x - self.panel_rect.x, y - self.panel_rect.y)
            return True, False
        return False, False

    def handle_motion(self, pos):
        if self.dragging and self.panel_rect is not None:
            self.pos = (pos[0] - self.drag_offset[0], pos[1] - self.drag_offset[1])
            self.user_moved = True
            return True
        return False

    def handle_release(self):
        was_dragging = self.dragging
        self.dragging = False
        return was_dragging

    def handle_wheel(self, dy):
        if not self.active or self.collapsed:
            return False
        self.scroll = max(0, self.scroll - max(1, dy))
        return True

    def handle_key(self, key):
        if not self.active:
            return False
        if key == pygame.K_LEFT:
            self.tab = (self.tab - 1) % 3
            self.scroll = 0
            return True
        if key == pygame.K_RIGHT:
            self.tab = (self.tab + 1) % 3
            self.scroll = 0
            return True
        if key in (pygame.K_UP, pygame.K_DOWN):
            self.scroll = max(0, self.scroll + (3 if key == pygame.K_DOWN else -3))
            return True
        return False

    def contains(self, pos):
        return self.panel_rect is not None and self.panel_rect.collidepoint(pos)

    def _copy_coord(self, key, state):
        fmt = state.get("point_output_format", "a+bi")
        m_pt = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
        j_pt = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
        c_pt = state.get("julia_c", DEFAULT_JULIA_C)
        if key in ("coord-copy-m", "m"):
            point = m_pt
        elif key in ("coord-copy-j", "j"):
            point = j_pt
        elif key in ("coord-copy-c", "c"):
            point = c_pt
        elif key == "all":
            return self._copy_text(
                f"M: {self._fmt(m_pt, fmt)}\n"
                f"J: {self._fmt(j_pt, fmt)}\n"
                f"C: {self._fmt(c_pt, fmt)}")
        else:
            return None
        return self._copy_text(self._fmt(point, fmt))

    def _copy_text(self, text):
        """Copy text on a daemon thread so the render loop never blocks.

        Returns the worker thread so callers/tests can join it if needed.
        """
        def worker():
            ok = _set_clipboard(text)
            self._copy_feedback = (bool(ok), pygame.time.get_ticks())
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _tab_lines(self, state, data, tab, fmt):
        """Content lines for one Points tab, colored from the points palette."""
        accent = _palette_section_color(state, "points menu", 1.0)
        label = _palette_section_color(state, "points menu", 0.85)
        dim = (170, 170, 185)
        lines = []

        def add(text, color=None):
            lines.append((text, False, color if color is not None else label))

        def add_cycle_lines(cls, center=None):
            if cls.get("proven"):
                add(f"cycle length: {cls.get('cycle_len', 0)}")
                for i, cp in enumerate(cls.get("cycle_pts", [])[:8]):
                    add(f"  cycle[{i}]: {self._fmt(cp, fmt)}")
                m = cls.get("multiplier")
                if m is not None:
                    add(f"multiplier: {self._fmt((m.real, m.imag), fmt)}")
                add(f"log|mult|: {cls.get('log_mult', 'n/a')}")
            else:
                add("cycle length: 0 (unproven)")
                cand = cls.get("candidate_len", 0)
                if cand:
                    add(f"candidate period: {cand}")
                res = cls.get("residual")
                if res is not None:
                    add(f"residual: {res:.3e}")
            if center is not None:
                add(f"center point: {self._fmt((center.real, center.imag), fmt)}")
            add(f"confidence: {cls.get('confidence', 0)}%", dim)
            add(f"evidence: {cls.get('evidence', 'n/a')}", dim)

        if tab == 0:
            cls = data["cls_m"]
            add("Mandelbrot orbit point", accent)
            add(f"type: {cls.get('status', '?')}")
            add(f"coord: {self._fmt(data['orbit_pt_m'], fmt)}")
            add_cycle_lines(cls, data.get("center_m"))
            add(f"pixel: {data['mb_px'][0]}, {data['mb_px'][1]}", dim)
        elif tab == 1:
            cls = data["cls_j"]
            add("Julia orbit point", accent)
            add(f"type: {cls.get('status', '?')}")
            add(f"coord: {self._fmt(data['orbit_pt_j'], fmt)}")
            add_cycle_lines(cls, data.get("center_j"))
            add(f"pixel: {data['ju_px'][0]}, {data['ju_px'][1]}", dim)
        else:
            add("Julia constant c", accent)
            add(f"coord: {self._fmt(data['julia_c'], fmt)}")
            add(f"pixel: {data['c_px'][0]}, {data['c_px'][1]}", dim)
            add(f"format: {fmt}", dim)
        return lines

    def draw(self, screen, font, state, xmin, xmax, ymin, ymax, width=None,
             height=None, pane_bounds=None):
        if not self.active:
            return
        if width is None:
            width = screen.get_width()
        if height is None:
            height = screen.get_height()
        data = self._gather_data(state, xmin, xmax, ymin, ymax, width, height, pane_bounds)
        scale = max(0.5, min(1.5, min(width / 1920, height / 1080)))
        fs = max(10, int(13 * scale))
        sfont = pygame.font.SysFont("monospace", fs)
        pad = max(6, int(8 * scale))
        line_h = sfont.get_height() + max(3, int(4 * scale))
        pad_inner = max(4, int(6 * scale))

        fmt = data["fmt"]
        content = self._tab_lines(state, data, self.tab, fmt)

        th = sfont.get_height()
        s = lambda v: max(1, int(v * scale))

        tabs = [("M", "MB orbit"), ("J", "Julia orbit"), ("C", "Julia c")]
        btn_h = max(s(22), th)
        tab_w = max(s(80), max(sfont.size(name)[0] for _, name in tabs) + s(14))
        copy_label = ("Copy M", "Copy J", "Copy C")[self.tab]
        copy_w = max(s(70), sfont.size(copy_label)[0] + s(16))
        fmt_label = "a+bi" if fmt == "a+bi" else "(x, y)"
        fmt_w = max(s(64), sfont.size(fmt_label)[0] + s(16))
        content_w = max((sfont.size(t)[0] for t, _, _ in content), default=s(200)) + 2 * pad_inner
        bottom_w = tab_w * 3 + s(6) * 2 + fmt_w + s(6) + copy_w + s(12)
        panel_w = min(width - 2 * pad, max(content_w, bottom_w) + 2 * pad_inner)
        max_panel_h = height - 2 * pad
        visible_lines = max(1, (max_panel_h - btn_h - 2 * pad_inner) // line_h)
        shown = [] if self.collapsed else content[self.scroll:self.scroll + visible_lines]
        # Only the tab row is shared, so the bottom anchor keeps it from moving.
        panel_h = btn_h + 2 * pad_inner
        if not self.collapsed:
            panel_h += len(shown) * line_h
        panel_h = min(max_panel_h, panel_h)

        if self.user_moved and self.pos is not None:
            sx, sy = self.pos
        else:
            # Keep the bottom edge fixed so the shared tab row never moves.
            sx, sy = pad, max(pad, height - panel_h - pad)
        sx = max(0, min(max(0, width - panel_w), int(sx)))
        sy = max(0, min(max(0, height - panel_h), int(sy)))

        bg = _palette_section_color(state, "points menu", 0.0)
        border = _palette_section_color(state, "points menu", 1.0)
        accent = _palette_section_color(state, "points menu", 0.75)
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((*bg, 230))
        pygame.draw.rect(surf, border, (0, 0, panel_w, panel_h), max(1, s(2)))

        self.button_rects = {}
        content_top = pad_inner
        tab_row_y = panel_h - pad_inner - btn_h
        content_h = max(line_h, tab_row_y - content_top - s(4))
        if self.collapsed:
            col = sfont.render("(collapsed)", True, (170, 170, 185))
            surf.blit(col, (pad_inner, content_top))
        else:
            yy = content_top
            for text, _bold, color in shown:
                if yy + line_h > content_top + content_h:
                    break
                if text:
                    ts = sfont.render(text, True, color or (220, 220, 220))
                    surf.blit(ts, (pad_inner, yy))
                yy += line_h
            if len(content) > visible_lines:
                bar_x = panel_w - s(6)
                frac = max(0.1, visible_lines / len(content))
                knob_h = max(s(10), int(content_h * frac))
                max_scroll = max(1, len(content) - visible_lines)
                knob_y = content_top + int((content_h - knob_h) * (self.scroll / max_scroll))
                pygame.draw.rect(surf, (70, 70, 90), (bar_x, content_top, s(4), content_h), 0, s(2))
                pygame.draw.rect(surf, accent, (bar_x, knob_y, s(4), knob_h), 0, s(2))

        bx = pad_inner
        for i, (_short, name) in enumerate(tabs):
            row = pygame.Rect(bx, tab_row_y, tab_w, btn_h)
            active = (i == self.tab) and not self.collapsed
            pygame.draw.rect(surf, accent if active else (40, 40, 52), row, 0, s(3))
            pygame.draw.rect(surf, border, row, s(1))
            label = sfont.render(name, True, (255, 255, 255))
            surf.blit(label, (row.x + (row.w - label.get_width()) // 2,
                              row.y + (btn_h - th) // 2))
            self.button_rects[f"tab-{i}"] = (sx + row.x, sy + row.y, row.w, row.h)
            bx += tab_w + s(6)
        copy_rect = pygame.Rect(panel_w - pad_inner - copy_w, tab_row_y, copy_w, btn_h)
        fmt_rect = pygame.Rect(copy_rect.x - s(6) - fmt_w, tab_row_y, fmt_w, btn_h)

        pygame.draw.rect(surf, (50, 50, 62), fmt_rect, 0, s(3))
        pygame.draw.rect(surf, border, fmt_rect, s(1))
        ft = sfont.render(fmt_label, True, (220, 220, 220))
        surf.blit(ft, (fmt_rect.x + (fmt_rect.w - ft.get_width()) // 2,
                       fmt_rect.y + (btn_h - th) // 2))
        self.button_rects["format"] = (sx + fmt_rect.x, sy + fmt_rect.y,
                                       fmt_rect.w, fmt_rect.h)

        pygame.draw.rect(surf, (50, 50, 62), copy_rect, 0, s(3))
        pygame.draw.rect(surf, border, copy_rect, s(1))
        ct = sfont.render(copy_label, True, (255, 255, 255))
        surf.blit(ct, (copy_rect.x + (copy_rect.w - ct.get_width()) // 2,
                       copy_rect.y + (btn_h - th) // 2))
        self.button_rects["copy-tab"] = (sx + copy_rect.x, sy + copy_rect.y,
                                         copy_rect.w, copy_rect.h)

        fb = self._copy_feedback
        if fb is not None and pygame.time.get_ticks() - fb[1] < 1200:
            msg = "copied" if fb[0] else "copy failed"
            col = (120, 230, 120) if fb[0] else (240, 120, 120)
            fbt = sfont.render(msg, True, col)
            surf.blit(fbt, (max(pad_inner, fmt_rect.x - fbt.get_width() - s(6)),
                            fmt_rect.y + (btn_h - th) // 2))

        screen.blit(surf, (sx, sy))
        self.panel_rect = pygame.Rect(sx, sy, panel_w, panel_h)
        self.content_rect = pygame.Rect(sx + pad_inner, sy + content_top,
                                        max(s(40), panel_w - 2 * pad_inner - s(8)),
                                        max(s(20), content_h))


def _complex_to_screen_pane(cx, cy, pane=None, width=0, height=0,
                            xmin=0, xmax=0, ymin=0, ymax=0):
    """Convert complex-plane (cx, cy) to screen pixel (px, py).

    When pane is provided (split mode), uses that pane's offset and
    aspect-correct view bounds.  Otherwise uses full-window bounds.
    """
    if pane is not None:
        px0 = pane["x"]
        py0 = pane["y"]
        pw = pane["w"]
        ph = pane["h"]
        bxmin = pane["p_xmin"]
        bxmax = pane["p_xmax"]
        bymin = pane["p_ymin"]
        bymax = pane["p_ymax"]
    else:
        px0, py0, pw, ph = 0, 0, width, height
        bxmin, bxmax = xmin, xmax
        bymin, bymax = ymin, ymax
    if (bxmax - bxmin) > 0:
        sx = px0 + (cx - bxmin) / (bxmax - bxmin) * pw
    else:
        sx = px0 + pw // 2
    if (bymax - bymin) > 0:
        sy = py0 + (bymax - cy) / (bymax - bymin) * ph
    else:
        sy = py0 + ph // 2
    return int(sx), int(sy)


def _orbit_pixel_color(i, smooth, colortable, ncycle):
    """Map orbit iteration index to a color using the same colormap as the graph."""
    ncol = colortable.shape[0] - 1
    if smooth:
        cniter = min(1.0, math.sqrt(i + 1) / ncycle)
        col_i = round(cniter * ncol)
    else:
        sqrt_n = math.sqrt(i + 1)
        floor_n = float(int(sqrt_n))
        cniter = floor_n / ncycle
        col_i = round(min(1.0, cniter) * ncol)
    r = colortable[col_i, 0]
    g = colortable[col_i, 1]
    b = colortable[col_i, 2]
    return (max(0, min(255, int(r * 255))),
            max(0, min(255, int(g * 255))),
            max(0, min(255, int(b * 255))))


def _draw_grid(screen, state, xmin, xmax, ymin, ymax, width, height,
               origin_x=0, origin_y=0, font=None):
    """Draw a zoom-adaptive grid overlay with configurable opacity.

    Grid spacing adapts to the current zoom level using nice intervals
    (powers of 10 or 1/2/5 multiples).  Major lines every spacing, minor
    lines every spacing/5 at deep zoom levels.  ``grid_opacity`` and
    ``grid_label_opacity`` are independent; either at 0 skips its work.
    """
    opacity = max(0.0, min(1.0, state.get("grid_opacity", DEFAULT_GRID_OPACITY)))
    label_opacity = max(0.0, min(1.0, state.get("grid_label_opacity", DEFAULT_GRID_LABEL_OPACITY)))
    draw_lines = opacity > 0.0
    draw_labels = label_opacity > 0.0 and font is not None
    if not draw_lines and not draw_labels:
        return

    spacing = _compute_grid_spacing(xmin, xmax, ymin, ymax, width, height)
    minor_spacing = spacing / 5.0
    start_x = math.floor(xmin / spacing) * spacing
    start_y = math.floor(ymin / spacing) * spacing
    inv_rx = width / (xmax - xmin) if (xmax - xmin) > 0 else 0
    inv_ry = height / (ymax - ymin) if (ymax - ymin) > 0 else 0

    if draw_lines:
        grid_val = int(255 * opacity)
        minor_val = int(255 * opacity * 0.35)
        color = (grid_val, grid_val, grid_val, grid_val)
        minor_color = (minor_val, minor_val, minor_val, minor_val)

        grid_surf = pygame.Surface((width, height), pygame.SRCALPHA)

        x = start_x
        while x <= xmax:
            px = int(origin_x + (x - xmin) * inv_rx)
            if 0 <= px <= origin_x + width:
                pygame.draw.line(grid_surf, minor_color, (px, 0),
                                 (px, height), 1)
            x += minor_spacing
        y = start_y
        while y <= ymax:
            py = int(origin_y + (ymax - y) * inv_ry)
            if 0 <= py <= origin_y + height:
                pygame.draw.line(grid_surf, minor_color, (0, py),
                                 (width, py), 1)
            y += minor_spacing

        x = start_x
        while x <= xmax:
            px = int(origin_x + (x - xmin) * inv_rx)
            if 0 <= px <= origin_x + width:
                pygame.draw.line(grid_surf, color, (px, 0), (px, height), 1)
            x += spacing
        y = start_y
        while y <= ymax:
            py = int(origin_y + (ymax - y) * inv_ry)
            if 0 <= py <= origin_y + height:
                pygame.draw.line(grid_surf, color, (0, py), (width, py), 1)
            y += spacing
        screen.blit(grid_surf, (origin_x, origin_y))

    if not draw_labels:
        return
    label_color = _color_opacity(_palette_section_color(state, "grid", 1.0), label_opacity)

    x = start_x
    while x <= xmax:
        px = int(origin_x + (x - xmin) * inv_rx)
        if origin_x <= px <= origin_x + width:
            ts = pygame.transform.rotate(font.render(_grid_label_text(x, spacing), True, label_color), 90)
            tx = px - ts.get_width() // 2
            ty = origin_y + height - ts.get_height() - 2
            if origin_y <= ty <= origin_y + height - ts.get_height():
                screen.blit(ts, (tx, ty))
        x += spacing

    y = start_y
    while y <= ymax:
        py = int(origin_y + (ymax - y) * inv_ry)
        if origin_y <= py <= origin_y + height:
            ts = font.render(_grid_label_text(y, spacing), True, label_color)
            ty = py - ts.get_height() // 2
            if origin_y <= ty <= origin_y + height - ts.get_height():
                screen.blit(ts, (origin_x + 3, ty))
        y += spacing


def _resolve_mb_ju_panes(pane_bounds):
    """Return (mandelbrot_pane, julia_pane) for a split pane list."""
    mb_pane = pane_bounds[0] if not pane_bounds[0]["is_julia"] else pane_bounds[1]
    ju_pane = pane_bounds[1] if mb_pane is pane_bounds[0] else pane_bounds[0]
    return mb_pane, ju_pane


def _point_marker_color(state, section_key):
    """Marker color from a palette section, inverted during a highlight flash."""
    color = _palette_section_color(state, section_key)
    now = pygame.time.get_ticks()
    if state.get("highlight_points", False):
        period = max(1, int(state.get("highlight_flash_duration_ms",
                                        DEFAULT_HIGHLIGHT_FLASH_DURATION_MS)))
        elapsed = now - state.get("highlight_flash_time", 0)
        freq = state.get("highlight_flash_frequency", DEFAULT_HIGHLIGHT_FLASH_FREQUENCY)
        if 0 <= elapsed < period and math.sin(elapsed * freq * 6.2831853) >= 0:
            color = (255 - color[0], 255 - color[1], 255 - color[2])
    return color


def _draw_point_markers(screen, state, xmin, xmax, ymin, ymax, width, height,
                        pane_bounds=None):
    """Draw the M/J/C markers, colored from their palette sections.

    Each marker's opacity is its on/off switch; a marker at 0 opacity is not
    colored or projected.
    """
    in_split = pane_bounds is not None and len(pane_bounds) > 1
    if in_split:
        mb_pane, ju_pane = _resolve_mb_ju_panes(pane_bounds)
    else:
        mb_pane = ju_pane = None
    julia_visible = in_split or state.get("set_blend", 0.0) > 0.0

    m_opacity = max(0.0, min(1.0, state.get("point_opacity_m", 1.0)))
    j_opacity = max(0.0, min(1.0, state.get("point_opacity_j", 1.0)))
    c_opacity = max(0.0, min(1.0, state.get("point_opacity_c", 1.0)))
    if m_opacity <= 0.0 and not (julia_visible and (j_opacity > 0.0 or c_opacity > 0.0)):
        return

    drag_mode = state.get("orbit_drag_mode")
    highlight = bool(state.get("highlight_points", False))

    def _draw(cx, cy, pane, color, size, opacity, ring=False):
        color = _color_opacity(color, opacity)
        px, py = _complex_to_screen_pane(cx, cy, pane=pane, width=width, height=height,
                                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        pygame.draw.circle(screen, color, (px, py), size)
        if ring or highlight:
            pygame.draw.circle(screen, (200, 200, 200), (px, py), size + 2, 1)

    if m_opacity > 0.0:
        mx, my = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
        _draw(mx, my, mb_pane, _point_marker_color(state, "mb orbit point"),
              max(1, int(state.get("point_size_m", state.get("point_size", DEFAULT_POINT_SIZE)))),
              m_opacity, ring=(drag_mode == "mandelbrot"))
    if julia_visible and j_opacity > 0.0:
        jx, jy = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
        _draw(jx, jy, ju_pane, _point_marker_color(state, "ju orbit point"),
              max(1, int(state.get("point_size_j", state.get("point_size", DEFAULT_POINT_SIZE)))),
              j_opacity, ring=(drag_mode == "julia"))
    if julia_visible and c_opacity > 0.0:
        cx, cy = state.get("julia_c", DEFAULT_JULIA_C)
        _draw(cx, cy, mb_pane, _point_marker_color(state, "julia c point"),
              max(1, int(state.get("point_size_c", state.get("point_size", DEFAULT_POINT_SIZE)))),
              c_opacity, ring=(drag_mode == "julia_c"))


def _fixed_point_candidates_for_c(cx, cy):
    """Return the two period-1 fixed-point candidates of z -> z^2 + c.

    These are the roots of z^2 - z + c = 0, i.e. math.txt `Z_pos` / `Z_neg`.
    Only one of them is the orbit's actual center; see `_center_point_for`.
    """
    root = (complex(1.0 - 4.0 * cx, -4.0 * cy)) ** 0.5
    return ((1.0 + root) / 2.0, (1.0 - root) / 2.0)


def _center_point_for(cx, cy, cls):
    """Select the correct period-1 center from the two candidates.

    math.txt `Fixed Points` picks whichever of Z_pos / Z_neg is nearest to
    ``mean(C_ML)``, the average of the orbit/cycle points. `classify_orbit`
    reports that average as ``mean_pt``; fall back to the first candidate only
    when the classification is empty.
    """
    candidates = _fixed_point_candidates_for_c(cx, cy)
    avg = (cls or {}).get("mean_pt") or (0.0, 0.0)
    ax, ay = float(avg[0]), float(avg[1])

    def _dist2(z):
        return (z.real - ax) ** 2 + (z.imag - ay) ** 2

    return min(candidates, key=_dist2)


def _draw_cycle_center_points(screen, state, xmin, xmax, ymin, ymax, width, height,
                             pane_bounds=None):
    """Draw cycle points and the selected period-1 center point for the M and J orbits.

    Cycle/center opacity are the on/off switches; when both are 0 the orbits
    are not classified.
    """
    cycle_opacity = max(0.0, min(1.0, state.get("cycle_opacity", DEFAULT_CYCLE_OPACITY)))
    center_opacity = max(0.0, min(1.0, state.get("center_opacity", DEFAULT_CENTER_OPACITY)))
    if cycle_opacity <= 0.0 and center_opacity <= 0.0:
        return
    draw_cycle = cycle_opacity > 0.0
    draw_center = center_opacity > 0.0
    in_split = pane_bounds is not None and len(pane_bounds) > 1
    if in_split:
        mb_pane, ju_pane = _resolve_mb_ju_panes(pane_bounds)
    else:
        mb_pane = ju_pane = None

    classify_iter = state.get("orbit_max_iter", DEFAULT_ORBIT_MAX_ITER)
    julia_c = state.get("julia_c", DEFAULT_JULIA_C)
    julia_visible = in_split or state.get("set_blend", 0.0) > 0.0
    cycle_color = _color_opacity(_palette_section_color(state, "cycles"), cycle_opacity)
    center_color = _color_opacity(_palette_section_color(state, "center"), center_opacity)
    cycle_size = max(2, int(state.get("cycle_size", DEFAULT_CYCLE_SIZE)))
    center_size = max(2, int(state.get("center_size", DEFAULT_CENTER_SIZE)))

    def _draw_complex(cx, cy, pane, color, radius, diamond=False):
        px, py = _complex_to_screen_pane(cx, cy, pane=pane, width=width, height=height,
                                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        if diamond:
            pygame.draw.polygon(screen, color, [
                (px, py - radius), (px + radius, py),
                (px, py + radius), (px - radius, py)])
        else:
            pygame.draw.circle(screen, color, (px, py), radius)

    def _draw_set(sx, sy, use_julia, pane, c_re, c_im):
        if pane is None and in_split:
            return
        cls = _classify_cached(sx, sy, use_julia, c_re, c_im, classify_iter)
        if draw_cycle and cls.get("proven"):
            for pt in cls.get("cycle_pts", []):
                _draw_complex(pt[0], pt[1], pane, cycle_color, cycle_size)
        if draw_center:
            center = _center_point_for(c_re, c_im, cls)
            _draw_complex(center.real, center.imag, pane, center_color, center_size, diamond=True)

    mx, my = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
    _draw_set(mx, my, False, mb_pane, mx, my)
    if julia_visible:
        jx, jy = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
        _draw_set(jx, jy, True, ju_pane, julia_c[0], julia_c[1])


def _draw_orbit(screen, state, xmin, xmax, ymin, ymax, width, height,
                pane_bounds=None):
    """Draw both orbits and the point/cycle markers.

    In split mode each orbit is confined to its own pane.  Orbit dots and
    lines each have their own per-set opacity; a set whose dots and lines are
    both 0 is not computed.
    """
    in_split = pane_bounds is not None and len(pane_bounds) > 1

    max_iter = state.get("orbit_max_iter", DEFAULT_ORBIT_MAX_ITER)
    julia_c = state.get("julia_c", DEFAULT_JULIA_C)

    set_blend = state.get("set_blend", 0.0)
    mb_alpha = 1.0 - set_blend
    ju_alpha = set_blend
    if in_split:
        mb_alpha = 1.0
        ju_alpha = 1.0

    mb_dots = mb_alpha * max(0.0, min(1.0, state.get("orbits_opacity_m", DEFAULT_ORBITS_OPACITY_M)))
    mb_lines = mb_alpha * max(0.0, min(1.0, state.get("lines_opacity_m", DEFAULT_LINES_OPACITY_M)))
    ju_dots = ju_alpha * max(0.0, min(1.0, state.get("orbits_opacity_j", DEFAULT_ORBITS_OPACITY_J)))
    ju_lines = ju_alpha * max(0.0, min(1.0, state.get("lines_opacity_j", DEFAULT_LINES_OPACITY_J)))
    draw_mb = mb_dots > 0.0 or mb_lines > 0.0
    draw_ju = ju_dots > 0.0 or ju_lines > 0.0

    orbit_pt_m = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
    orbit_pt_j = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)

    if draw_mb or draw_ju:
        gradient_stops = state.get("gradient_stops") or DEFAULT_GRADIENT_STOPS
        gradient_blend = _active_gradient_blend(state)
        orbit_cache_key = ("gradient", tuple(s[0] for s in gradient_stops),
                           tuple(tuple(s[1]) for s in gradient_stops), gradient_blend)
        if orbit_cache_key not in _ORBIT_COLOR_CACHE:
            _ORBIT_COLOR_CACHE[orbit_cache_key] = _make_gradient_colortable(
                gradient_stops, gradient_blend)
            if len(_ORBIT_COLOR_CACHE) > _ORBIT_COLOR_CACHE_MAX:
                _ORBIT_COLOR_CACHE.clear()
        colortable = _ORBIT_COLOR_CACHE[orbit_cache_key]

        ncycle = math.sqrt(max_iter)
        smooth = state.get("smooth", True)
        pt_size = state.get("orbit_point_size", 3)

        def _draw_single_orbit(sx, sy, use_julia, c_re, c_im, dots_opacity, lines_opacity,
                               px0=0, py0=0, pw=None, ph=None,
                               bxmin=None, bxmax=None, bymin=None, bymax=None):
            if dots_opacity <= 0.0 and lines_opacity <= 0.0:
                return
            if pw is None:
                pw = width
            if ph is None:
                ph = height
            if bxmin is None:
                bxmin, bxmax, bymin, bymax = xmin, xmax, ymin, ymax
            points = compute_orbit(sx, sy, max_iter, use_julia, c_re, c_im)
            orb_key = (sx, sy, max_iter, use_julia, c_re, c_im)
            _cached_orb = _ORBIT_CACHE.get(orb_key)
            if _cached_orb is None:
                if len(_ORBIT_CACHE) > 256:
                    _ORBIT_CACHE.clear()
                _cached_orb = points
                _ORBIT_CACHE[orb_key] = points
            points = _cached_orb
            screen_pts = _mandelbrot_to_screen_all(points, bxmin, bxmax, bymin, bymax, pw, ph)

            if lines_opacity > 0.0:
                lines_ct = _palette_section_colortable_or(
                    state, "ju lines" if use_julia else "mb lines", colortable)
                for i in range(len(screen_pts) - 1):
                    color = _orbit_pixel_color(i + 1, smooth, lines_ct, ncycle)
                    color = (max(0, min(255, int(color[0] * lines_opacity))),
                             max(0, min(255, int(color[1] * lines_opacity))),
                             max(0, min(255, int(color[2] * lines_opacity))))
                    clipped = _liang_barsky_clip(
                        screen_pts[i][0], screen_pts[i][1],
                        screen_pts[i + 1][0], screen_pts[i + 1][1],
                        0, 0, pw, ph)
                    if clipped:
                        pygame.draw.line(screen, color,
                                         (px0 + clipped[0], py0 + clipped[1]),
                                         (px0 + clipped[2], py0 + clipped[3]), 1)

            if dots_opacity > 0.0:
                dots_ct = _palette_section_colortable_or(
                    state, "ju orbits" if use_julia else "mb orbits", colortable)
                for i in range(len(screen_pts)):
                    if 0 <= screen_pts[i][0] <= pw and 0 <= screen_pts[i][1] <= ph:
                        color = _orbit_pixel_color(i, smooth, dots_ct, ncycle)
                        color = (max(0, min(255, int(color[0] * dots_opacity))),
                                 max(0, min(255, int(color[1] * dots_opacity))),
                                 max(0, min(255, int(color[2] * dots_opacity))))
                        radius = max(1, int(pt_size * (0.5 + 0.5 * i / len(screen_pts))))
                        pygame.draw.circle(screen, color,
                                           (px0 + int(screen_pts[i][0]), py0 + int(screen_pts[i][1])), radius)

        if in_split:
            mb_pane = pane_bounds[0] if not pane_bounds[0]["is_julia"] else pane_bounds[1]
            ju_pane = pane_bounds[0] if pane_bounds[0]["is_julia"] else pane_bounds[1]

            if draw_mb:
                _draw_single_orbit(orbit_pt_m[0], orbit_pt_m[1], False,
                                   julia_c[0], julia_c[1], mb_dots, mb_lines,
                                   px0=mb_pane["x"], py0=mb_pane["y"],
                                   pw=mb_pane["w"], ph=mb_pane["h"],
                                   bxmin=mb_pane["p_xmin"], bxmax=mb_pane["p_xmax"],
                                   bymin=mb_pane["p_ymin"], bymax=mb_pane["p_ymax"])

            if draw_ju:
                _draw_single_orbit(orbit_pt_j[0], orbit_pt_j[1], True,
                                   julia_c[0], julia_c[1], ju_dots, ju_lines,
                                   px0=ju_pane["x"], py0=ju_pane["y"],
                                   pw=ju_pane["w"], ph=ju_pane["h"],
                                   bxmin=ju_pane["p_xmin"], bxmax=ju_pane["p_xmax"],
                                   bymin=ju_pane["p_ymin"], bymax=ju_pane["p_ymax"])

        else:
            if draw_mb:
                _draw_single_orbit(orbit_pt_m[0], orbit_pt_m[1], False,
                                   julia_c[0], julia_c[1], mb_dots, mb_lines)

            if draw_ju:
                _draw_single_orbit(orbit_pt_j[0], orbit_pt_j[1], True,
                                   julia_c[0], julia_c[1], ju_dots, ju_lines)

    _draw_cycle_center_points(screen, state, xmin, xmax, ymin, ymax, width, height,
                             pane_bounds=pane_bounds)
    _draw_point_markers(screen, state, xmin, xmax, ymin, ymax, width, height,
                        pane_bounds=pane_bounds)


def compute_image(width, height, xmin, xmax, ymin, ymax, maxiter, params,
                  mb_params=None, ju_alpha=0.0):
    """Render a single set or blend two sets.

    If mb_params is None or ju_alpha is 0.0: render single set from params.
    If mb_params is provided and ju_alpha > 0: render Julia from params (ju_alpha),
    blend with Mandelbrot from mb_params (1 - ju_alpha).
    """
    if mb_params is None or ju_alpha <= 0.0:
        return _render_single(width, height, xmin, xmax, ymin, ymax, maxiter, params)

    mb_img = _render_single(width, height, xmin, xmax, ymin, ymax, maxiter, mb_params)
    ju_img = _render_single(width, height, xmin, xmax, ymin, ymax, maxiter, params)
    mb_alpha = 1.0 - ju_alpha
    mb_img = mb_img.astype(np.float64)
    ju_img = ju_img.astype(np.float64)
    blended = mb_img * mb_alpha + ju_img * ju_alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def _render_single(width, height, xmin, xmax, ymin, ymax, maxiter, params):
    """Render a single set (Mandelbrot or Julia) to a uint8 array."""
    colortable = params["colortable"]
    ncycle = params["ncy"]
    stripe_s = params["stripe_s"]
    stripe_sig = params["stripe_sig"]
    step_s = params["step_s"]
    light = params["light"]
    smooth = params.get("smooth", True)
    use_julia = params.get("use_julia", False)
    julia_c_re = params.get("julia_c_re", 0.0)
    julia_c_im = params.get("julia_c_im", 0.0)
    view_bounds = params.get("view_bounds") or (xmin, xmax, ymin, ymax)
    precision_mode, precision_bits = _resolve_precision(
        params.get("precision_mode", DEFAULT_PRECISION_MODE),
        params.get("precision_bits", DEFAULT_PRECISION_BITS),
        view_bounds)
    tier = _select_precision_tier(*view_bounds, mode=precision_mode)
    params["precision_tier"] = tier
    params["precision_bits"] = precision_bits
    params["last_uncertainty"] = None
    render_xmin = _as_float(xmin)
    render_xmax = _as_float(xmax)
    render_ymin = _as_float(ymin)
    render_ymax = _as_float(ymax)

    if tier == "mpfr":
        x_values = _mpfr_grid(view_bounds[0], view_bounds[1], width, precision_bits)
        y_values = _mpfr_grid(view_bounds[2], view_bounds[3], height, precision_bits)
        if x_values is None or y_values is None:
            warnings.warn("gmpy2 unavailable; high-precision render fell back to float64",
                          RuntimeWarning)
            tier = "float64"
        else:
            with localcontext() as ctx:
                ctx.prec = max(80, precision_bits)
                dx = _as_decimal(view_bounds[1]) - _as_decimal(view_bounds[0])
                dy = _as_decimal(view_bounds[3]) - _as_decimal(view_bounds[2])
                diag_decimal = (dx * dx + dy * dy).sqrt()
            diag = gmpy2.mpfr(str(diag_decimal), precision_bits)
            mat, uncertain = compute_set_mpfr(
                x_values, y_values, maxiter, colortable, ncycle,
                stripe_s, stripe_sig, step_s, diag, light,
                smooth=smooth, use_julia=use_julia,
                julia_c_re=julia_c_re, julia_c_im=julia_c_im,
                precision_bits=precision_bits,
                show_uncertainty=params.get("show_uncertainty", DEFAULT_SHOW_UNCERTAINTY))
            params["precision_backend"] = "mpfr"
            params["last_uncertainty"] = uncertain[::-1]
            if _RENDER_PROFILER:
                _RENDER_PROFILER.record_render(0, 0, 1)
            return _post_process(mat, params.get("fxaa", False))

    if tier == "float64" and params.get("use_gpu") and cuda is not None:
        try:
            if "_colortable_d" not in params:
                params["_colortable_d"] = cuda.to_device(colortable)
            if "_light_d" not in params:
                params["_light_d"] = cuda.to_device(light)
            colortable_d = params["_colortable_d"]
            light_d = params["_light_d"]

            npixels = width * height
            nthread = 512
            nblock = math.ceil(npixels / nthread)
            mat = cuda.device_array((height, width, 3), dtype=np.float32)
            compute_set_gpu[nblock, nthread](
                mat, int64(width), int64(height), int64(width), int64(height),
                int64(0), int64(0), float64(render_xmin), float64(render_xmax),
                float64(render_ymin), float64(render_ymax), int64(maxiter), colortable_d,
                float64(ncycle), float64(stripe_s), float64(stripe_sig),
                float64(step_s), float64(math.sqrt((render_xmin - render_xmax) ** 2 + (render_ymin - render_ymax) ** 2)), light_d, int64(smooth),
                int64(use_julia), float64(julia_c_re), float64(julia_c_im))
            cuda.synchronize()

            if _RENDER_PROFILER:
                _RENDER_PROFILER.record_render(0, 1, 0)

            result = mat.copy_to_host()
            params["precision_backend"] = "cuda"
            return _post_process(result, params.get("fxaa", False))
        except Exception as e:
            if os.environ.get("MB_DEBUG_PERF"):
                print(f"[gpu] falling back to CPU: {e}", file=sys.stderr)

    diag = math.sqrt((render_xmin - render_xmax) ** 2 + (render_ymin - render_ymax) ** 2)
    if tier == "perturbed":
        cx_ref = _as_float(_decimal_mid(view_bounds[0], view_bounds[1]))
        cy_ref = _as_float(_decimal_mid(view_bounds[2], view_bounds[3]))
        ref_orbit = (_get_ref_orbit_julia(cx_ref, cy_ref, maxiter, julia_c_re, julia_c_im)
                     if use_julia else _get_ref_orbit(cx_ref, cy_ref, maxiter))
        mat = compute_set_perturbed(
            np.linspace(_as_float(view_bounds[0]), _as_float(view_bounds[1]), width),
            np.linspace(_as_float(view_bounds[2]), _as_float(view_bounds[3]), height),
            ref_orbit, maxiter, colortable, ncycle,
            stripe_s, stripe_sig, step_s, diag, light,
            smooth, use_julia=use_julia,
            julia_c_re=julia_c_re, julia_c_im=julia_c_im)
        params["precision_backend"] = "perturbed-float64"
    else:
        mat = compute_set_cpu(np.linspace(_as_float(view_bounds[0]), _as_float(view_bounds[1]), width),
                              np.linspace(_as_float(view_bounds[2]), _as_float(view_bounds[3]), height),
                              maxiter, colortable, ncycle,
                              stripe_s, stripe_sig, step_s, diag, light,
                              smooth, use_julia, julia_c_re, julia_c_im)
        params["precision_backend"] = "float64"
    if _RENDER_PROFILER:
        _RENDER_PROFILER.record_render(0, 0, 1)
    return _post_process(mat, params.get("fxaa", False))


def _fix_aspect_ratio_decimal(x_min, x_max, y_min, y_max, width, height):
    if height <= 0 or width <= 0:
        return [x_min, x_max, y_min, y_max]
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range <= 0 or y_range <= 0:
        return [x_min, x_max, y_min, y_max]
    target_ratio = Decimal(width) / Decimal(height)
    current_ratio = x_range / y_range
    x_center = (x_min + x_max) / Decimal(2)
    y_center = (y_min + y_max) / Decimal(2)
    if current_ratio > target_ratio:
        new_y_range = x_range / target_ratio
        y_min = y_center - new_y_range / Decimal(2)
        y_max = y_center + new_y_range / Decimal(2)
    else:
        new_x_range = y_range * target_ratio
        x_min = x_center - new_x_range / Decimal(2)
        x_max = x_center + new_x_range / Decimal(2)
    return x_min, x_max, y_min, y_max


def fix_aspect_ratio(x_min, x_max, y_min, y_max, width, height):
    if height <= 0 or width <= 0:
        return [x_min, x_max, y_min, y_max]
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range <= 0 or y_range <= 0:
        return [x_min, x_max, y_min, y_max]
    target_ratio = width / height
    current_ratio = x_range / y_range
    if current_ratio > target_ratio:
        new_y_range = x_range / target_ratio
        y_center = (y_min + y_max) / 2
        y_min = y_center - new_y_range / 2
        y_max = y_center + new_y_range / 2
    else:
        new_x_range = y_range * target_ratio
        x_center = (x_min + x_max) / 2
        x_min = x_center - new_x_range / 2
        x_max = x_center + new_x_range / 2
    return x_min, x_max, y_min, y_max


def run_single_mode(settings):
    max_iter = get_persistent_setting(settings, "max-iterations", cast=int, default=100)
    x0 = get_runtime_input("x", prompt="Enter x coordinate")
    y0 = get_runtime_input("y", prompt="Enter y coordinate")
    orbit_x, orbit_y, escaped = iterate_single(x0, y0, max_iter)
    if escaped:
        print(f"Escaped after {len(orbit_x)} iterations")
    else:
        print(f"Bounded — ran full {max_iter} iterations without escaping")
    print("Last 5 points:")
    for i in range(max(0, len(orbit_x) - 5), len(orbit_x)):
        print(f"  ({orbit_x[i]:.6f}, {orbit_y[i]:.6f})")


_RENDER_CACHE: dict = {}
_RENDER_CACHE_MAX = 64
_SPLIT_PANE_CACHE: dict = {}
_SPLIT_PANE_CACHE_MAX = 32
_ORBIT_COLOR_CACHE: dict = {}
_ORBIT_COLOR_CACHE_MAX = 16
_ORBIT_CACHE: dict = {}
_CLASSIFY_CACHE: dict = {}
_CLASSIFY_CACHE_MAX = 128
_PERF_STATS = {"render_count": 0, "total_render_ms": 0.0}


def _canonical_bounds(bounds):
    return tuple(_as_decimal(value) for value in bounds)


def _render_cache_key(state, max_iter, use_julia, view_bounds=None):
    return (state.get("use_gpu", False) and _CUDA_AVAILABLE,
            max_iter, use_julia, tuple(state.get("julia_c", DEFAULT_JULIA_C)),
            state.get("smooth", True), state.get("fxaa", False),
            state.get("stripe_s", 0.0), state.get("stripe_sig", 0.9),
            state.get("step_s", 0.0), state.get("light_angle", DEFAULT_LIGHT_ANGLE),
            state.get("light_azim", DEFAULT_LIGHT_AZIM), state.get("light_i", DEFAULT_LIGHT_I),
            state.get("k_ambiant", DEFAULT_K_AMBIANT), state.get("k_diffuse", DEFAULT_K_DIFFUSE),
            state.get("k_specular", DEFAULT_K_SPECULAR), state.get("shininess", DEFAULT_SHININESS),
            tuple(state.get("gradient_stops")) if state.get("gradient_stops") else None,
            state.get("gradient_blend", DEFAULT_GRADIENT_BLEND),
            state.get("palette_index", 0),
            tuple(_palette_section_explicit(state, "ju set" if use_julia else "mb set") or ()),
            *_resolve_precision(state.get("precision_mode", DEFAULT_PRECISION_MODE),
                                state.get("precision_bits", DEFAULT_PRECISION_BITS),
                                view_bounds),
            bool(state.get("show_uncertainty", DEFAULT_SHOW_UNCERTAINTY)),
            _canonical_bounds(view_bounds) if view_bounds else None)


def _sync_julia_viewport(state, pane_bounds):
    """Sync state['julia_viewport'] with the corrected bounds from the Julia pane."""
    if pane_bounds is not None:
        for p in pane_bounds:
            if p["is_julia"]:
                state["julia_viewport"][0] = p["p_xmin"]
                state["julia_viewport"][1] = p["p_xmax"]
                state["julia_viewport"][2] = p["p_ymin"]
                state["julia_viewport"][3] = p["p_ymax"]
                break


def _compute_pane_bounds(split_mode, split_orientation,
                         xmin, xmax, ymin, ymax, width, height,
                         julia_viewport=None):
    """Compute pane dimensions and aspect-correct view bounds for split mode.

    Returns a list of dicts with keys:
      x, y, w, h         — pixel rectangle of the pane on screen
      p_xmin, p_xmax     — aspect-correct view bounds for this pane
      p_ymin, p_ymax
      is_julia           — True for Julia pane, False for Mandelbrot pane
    Returns None when split_mode is None.

    The Julia pane uses julia_viewport (independent dynamical-plane bounds)
    when provided, falling back to DEFAULT_JULIA_VIEWPORT.
    """
    if split_mode is None:
        return None

    if julia_viewport is None:
        julia_viewport = DEFAULT_JULIA_VIEWPORT
    ju_xmin, ju_xmax, ju_ymin, ju_ymax = julia_viewport

    if split_orientation == "vertical":
        pane_w = width // 2
        pane_h = height
        mb_bounds = fix_aspect_ratio(xmin, xmax, ymin, ymax, pane_w, pane_h)
        ju_bounds = fix_aspect_ratio(ju_xmin, ju_xmax, ju_ymin, ju_ymax, pane_w, pane_h)
        return [
            {"x": 0, "y": 0, "w": pane_w, "h": pane_h,
             "p_xmin": mb_bounds[0], "p_xmax": mb_bounds[1],
             "p_ymin": mb_bounds[2], "p_ymax": mb_bounds[3],
             "is_julia": False},
            {"x": pane_w, "y": 0, "w": pane_w, "h": pane_h,
             "p_xmin": ju_bounds[0], "p_xmax": ju_bounds[1],
             "p_ymin": ju_bounds[2], "p_ymax": ju_bounds[3],
             "is_julia": True},
        ]
    else:
        pane_w = width
        pane_h = height // 2
        mb_bounds = fix_aspect_ratio(xmin, xmax, ymin, ymax, pane_w, pane_h)
        ju_bounds = fix_aspect_ratio(ju_xmin, ju_xmax, ju_ymin, ju_ymax, pane_w, pane_h)
        return [
            {"x": 0, "y": 0, "w": pane_w, "h": pane_h,
             "p_xmin": mb_bounds[0], "p_xmax": mb_bounds[1],
             "p_ymin": mb_bounds[2], "p_ymax": mb_bounds[3],
             "is_julia": False},
            {"x": 0, "y": pane_h, "w": pane_w, "h": pane_h,
             "p_xmin": ju_bounds[0], "p_xmax": ju_bounds[1],
             "p_ymin": ju_bounds[2], "p_ymax": ju_bounds[3],
             "is_julia": True},
        ]


def render_to_surface(width, height, xmin, xmax, ymin, ymax, max_iter, state):
    """Render the set(s) to a pygame Surface.

    Uses set_blend from state to blend Mandelbrot (alpha = 1 - blend) and
    Julia (alpha = blend) sets. Skips rendering a set when its alpha is 0.

    In split mode, renders Mandelbrot and Julia panes side-by-side at
    full opacity each, ignoring set_blend.
    """
    split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)

    if split_mode is not None:
        orientation = state.get("split_orientation", DEFAULT_SPLIT_ORIENT)
        julia_vp = state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT)
        panes = _compute_pane_bounds(split_mode, orientation,
                                     xmin, xmax, ymin, ymax, width, height,
                                     julia_viewport=julia_vp)
        _sync_julia_viewport(state, panes)
        surf = pygame.Surface((width, height))
        surf.fill((0, 0, 0))
        used_gpu = False

        for pane in panes:
            if pane["w"] <= 0 or pane["h"] <= 0:
                continue
            use_julia = pane["is_julia"]
            pane_bounds = (pane["p_xmin"], pane["p_xmax"], pane["p_ymin"], pane["p_ymax"])
            key = _render_cache_key(state, max_iter, use_julia, pane_bounds)
            cached = _SPLIT_PANE_CACHE.get(key)
            cached_bounds = _canonical_bounds(pane_bounds) if cached is not None else None
            if cached is not None and cached[0] == cached_bounds and cached[1] == pane["w"] and cached[2] == pane["h"]:
                pane_surf = cached[3]
            else:
                params = _RENDER_CACHE.get(key)
                if params is None:
                    params = build_render_params(state, maxiter=max_iter, use_julia=use_julia,
                                                 view_bounds=pane_bounds)
                    _RENDER_CACHE[key] = params
                    if len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
                        _RENDER_CACHE.clear()
                rgb = _render_single(pane["w"], pane["h"],
                                     pane["p_xmin"], pane["p_xmax"],
                                     pane["p_ymin"], pane["p_ymax"],
                                     max_iter, params)
                pane_surf = pygame.surfarray.make_surface(rgb)
                used_gpu = used_gpu or params["use_gpu"]
                _SPLIT_PANE_CACHE[key] = (
                    cached_bounds, pane["w"], pane["h"], pane_surf)
                if len(_SPLIT_PANE_CACHE) > _SPLIT_PANE_CACHE_MAX:
                    _SPLIT_PANE_CACHE.clear()
            surf.blit(pane_surf, (pane["x"], pane["y"]))

        return surf, used_gpu and _CUDA_AVAILABLE

    set_blend = max(0.0, min(1.0, state.get("set_blend", 0.0)))
    mb_alpha = 1.0 - set_blend

    used_gpu = False

    mb_rgb = None
    if mb_alpha > 0.0:
        mb_key = _render_cache_key(state, max_iter, False, (xmin, xmax, ymin, ymax))
        mb_params = _RENDER_CACHE.get(mb_key)
        if mb_params is None:
            mb_params = build_render_params(state, maxiter=max_iter, use_julia=False,
                                            view_bounds=(xmin, xmax, ymin, ymax))
            _RENDER_CACHE[mb_key] = mb_params
            if len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
                _RENDER_CACHE.clear()
        mb_rgb = _render_single(width, height, xmin, xmax, ymin, ymax, max_iter, mb_params)
        used_gpu = mb_params["use_gpu"]

    if set_blend > 0.0:
        ju_key = _render_cache_key(state, max_iter, True, (xmin, xmax, ymin, ymax))
        ju_params = _RENDER_CACHE.get(ju_key)
        if ju_params is None:
            ju_params = build_render_params(state, maxiter=max_iter, use_julia=True,
                                            view_bounds=(xmin, xmax, ymin, ymax))
            _RENDER_CACHE[ju_key] = ju_params
            if len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
                _RENDER_CACHE.clear()
        ju_rgb = _render_single(width, height, xmin, xmax, ymin, ymax, max_iter, ju_params)
        used_gpu = used_gpu or ju_params["use_gpu"]

        if mb_rgb is not None:
            mb_f = mb_rgb.astype(np.float64)
            ju_f = ju_rgb.astype(np.float64)
            blended = mb_f * mb_alpha + ju_f * set_blend
            rgb_t = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            rgb_t = ju_rgb
    else:
        rgb_t = mb_rgb

    surf = pygame.surfarray.make_surface(rgb_t)
    return surf, used_gpu and _CUDA_AVAILABLE


def _render_perf_text(font, screen, lines):
    for i, line in enumerate(lines):
        surf = font.render(line, True, (200, 200, 200))
        screen.blit(surf, (10, 40 + i * 20))


class MenuOverlay:
    """On-screen menu overlay with clickable buttons for all incremental settings."""
    def __init__(self):
        self.active = False
        self.show_keybinds = False
        self.button_rects = {}
        self.menu_scale = 1.0
        self._keybind_cache = None
        self._menu_cache = None
        self.coord_input = None  # M J C or None - coordinate being edited
        self.coord_input_text = ""

    def toggle(self):
        self.active = not self.active

    def handle_click(self, pos, state):
        """Handle a mouse click on the menu. Returns (handled, needs_full_render)."""
        if not self.active or not self.button_rects:
            return False, False
        x, y = pos
        for key, rect_info in self.button_rects.items():
            rects = self._unpack_rects(rect_info, key)
            for r, rtype in rects:
                if r.collidepoint(x, y):
                    if key == "show-keybinds":
                        state["show_keybinds"] = not state.get("show_keybinds", False)
                        self.show_keybinds = state["show_keybinds"]
                        self._keybind_cache = None
                        self._menu_cache = None
                        return True, False
                    if key == "toggle-point-info":
                        state["show_point_info"] = not state.get("show_point_info", False)
                        self._menu_cache = None
                        return True, False
                    if key == "points-menu":
                        state["show_point_info"] = True
                        self._menu_cache = None
                        return True, False
                    if key == "keybinds-menu":
                        state["show_keybinds"] = True
                        self.show_keybinds = True
                        self._keybind_cache = None
                        self._menu_cache = None
                        return True, False
                    if key == "highlight-points":
                        state["highlight_points"] = True
                        state["highlight_flash_time"] = pygame.time.get_ticks()
                        self._menu_cache = None
                        return True, False
                    if key.startswith("coord-copy-"):
                        self._copy_coord(key, state)
                        return True, False
                    if key.startswith("coord-edit-"):
                        _label = key.replace("coord-edit-", "").upper()
                        if self.coord_input == _label:
                            changed = self._commit_coord_edit(state)
                            self.coord_input = None
                            self._menu_cache = None
                            self.coord_input_text = ""
                            if changed:
                                return True, True 
                            return True, False
                        else:
                            self._start_coord_edit(_label)
                            self._coord_input_text = ""
                            return True, False
                    self._do_action(key, rtype, state)
                    if key in ("cycle-palette", "load-palette-file"):
                        self._menu_cache = None
                    if key in ("grid_opacity", "grid_label_opacity", "auto_iter",
                               "orbits_opacity_m", "orbits_opacity_j",
                               "lines_opacity_m", "lines_opacity_j",
                               "point_opacity_m", "point_opacity_j", "point_opacity_c",
                               "cycle_opacity", "center_opacity", "highlight_points",
                               "reset-orbit-point-m", "reset-orbit-point-j"):
                        self._menu_cache = None
                        return True, False
                    return True, True
        self._menu_cache = None
        return False, False

    def _copy_coord(self, key, state):
        """Copy a point coordinate to clipboard in a+bi format."""
        coord_map = {
            "coord-copy-m": state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M),
            "coord-copy-j": state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J),
            "coord-copy-c": state.get("julia_c", DEFAULT_JULIA_C),
        }
        point = coord_map.get(key)
        if point is None:
            return
        xmin = state.get("xmin", -2.5)
        xmax = state.get("xmax", 1.0)
        ymin = state.get("ymin", -1.5)
        ymax = state.get("ymax", 1.5)
        text = MenuOverlay._format_complex(point[0], point[1], xmin, xmax, ymin, ymax)
        thread = threading.Thread(target=_set_clipboard, args=(text,), daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _format_complex(re, im, xmin, xmax, ymin, ymax):
        """Format (re, im) as a+bi with precision based on zoom level."""
        view_range = max(xmax - xmin, ymax - ymin) if xmax > xmin and ymax > ymin else 1.0
        decimals = max(6, min(15, int(-math.log10(view_range)) + 3)) if view_range > 0 else 6
        sign = "+" if im >= 0 else "-"
        return f"{re:.{decimals}f} {sign} {abs(im):.{decimals}f}i"

    def _start_coord_edit(self, label):
        """Start editing a coordinate point. label is 'M', 'J', or 'C'."""
        self.coord_input = label
        self.coord_input_text = ""

    def _commit_coord_edit(self, state):
        return _commit_coord_edit(state, self.coord_input, self.coord_input_text)

    def _unpack_rects(self, rect_info, key):
        """Convert stored button rects into (pygame.Rect, type) pairs."""
        if len(rect_info) == 4:
            return [(pygame.Rect(*rect_info), "click")]
        elif len(rect_info) == 8:
            return [
                (pygame.Rect(rect_info[0], rect_info[1], rect_info[2], rect_info[3]), "minus"),
                (pygame.Rect(rect_info[4], rect_info[5], rect_info[6], rect_info[7]), "plus"),
            ]
        elif len(rect_info) == 12:
            return [
                (pygame.Rect(rect_info[0], rect_info[1], rect_info[2], rect_info[3]), "minus"),
                (pygame.Rect(rect_info[4], rect_info[5], rect_info[6], rect_info[7]), "plus"),
                (pygame.Rect(rect_info[8], rect_info[9], rect_info[10], rect_info[11]), "reset"),
            ]
        return []

    def _do_action(self, key, rtype, state):
        """Perform the action associated with a button click.
        Normal click = 1x base step, SHIFT = 10x, CTRL = 0.1x. All floored to grid.
        Cyclic params wrap around instead of clamping."""
        mods = pygame.key.get_mods()
        is_shift = mods & (pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT)
        is_ctrl = mods & (pygame.KMOD_LCTRL | pygame.KMOD_RCTRL)
        cyclic_keys = {"light_angle", "light_azim", "light_i",
                       "k_ambiant", "k_diffuse", "k_specular"}
        rows_meta = {
            "max_iter":         (True,  state["max_iter"],         1,      RENDER_ITER_CAP, 1.0),
            "stripe_s":         (True,  int(state["stripe_s"]),     0,      100,  1.0),
            "step_s":           (True,  int(state["step_s"]),       0,      100,  1.0),
            "orbit_max_iter":   (True,  state["orbit_max_iter"],    1,      ORBIT_ITER_CAP,  10.0),
            "point_size_m":     (True,  max(1, int(state.get("point_size_m", DEFAULT_POINT_SIZE))), 1, 40, 1.0),
            "point_size_j":     (True,  max(1, int(state.get("point_size_j", DEFAULT_POINT_SIZE))), 1, 40, 1.0),
            "point_size_c":     (True,  max(1, int(state.get("point_size_c", DEFAULT_POINT_SIZE))), 1, 40, 1.0),
            "point_opacity_m":  (False, state.get("point_opacity_m", 1.0), 0.0, 1.0, 0.05),
            "point_opacity_j":  (False, state.get("point_opacity_j", 1.0), 0.0, 1.0, 0.05),
            "point_opacity_c":  (False, state.get("point_opacity_c", 1.0), 0.0, 1.0, 0.05),
            "cycle_opacity":    (False, state.get("cycle_opacity", DEFAULT_CYCLE_OPACITY), 0.0, 1.0, 0.05),
            "center_opacity":    (False, state.get("center_opacity", DEFAULT_CENTER_OPACITY), 0.0, 1.0, 0.05),
            "orbits_opacity_m": (False, state.get("orbits_opacity_m", DEFAULT_ORBITS_OPACITY_M), 0.0, 1.0, 0.05),
            "orbits_opacity_j": (False, state.get("orbits_opacity_j", DEFAULT_ORBITS_OPACITY_J), 0.0, 1.0, 0.05),
            "lines_opacity_m":  (False, state.get("lines_opacity_m", DEFAULT_LINES_OPACITY_M), 0.0, 1.0, 0.05),
            "lines_opacity_j":  (False, state.get("lines_opacity_j", DEFAULT_LINES_OPACITY_J), 0.0, 1.0, 0.05),
            "orbit_point_size": (True,  state.get("orbit_point_size", 3), 1,   10,   1.0),
            "c_point_size":     (True,  state.get("c_point_size", 5), 1,   20,   1.0),
            "shininess":        (False, float(state["shininess"]), 1.0,   100.0, 1.0),
            "light_angle":      (False, state["light_angle"],       0.0,   1.0,  0.1),
            "light_azim":       (False, state["light_azim"],        0.0,   1.0,  0.1),
            "light_i":          (False, state["light_i"],           0.0,   1.0,  0.01),
            "k_ambiant":        (False, state["k_ambiant"],         0.0,   1.0,  0.01),
            "k_diffuse":        (False, state["k_diffuse"],         0.0,   1.0,  0.01),
            "k_specular":       (False, state["k_specular"],        0.0,   1.0,  0.01),
            "set_blend":        (False, round(state.get("set_blend", 0.0), 4), 0.0, 1.0, 0.05),
            "grid_opacity":     (False, state.get("grid_opacity", 0.3), 0.0, 1.0, 0.05),
            "grid_label_opacity": (False, state.get("grid_label_opacity", DEFAULT_GRID_LABEL_OPACITY), 0.0, 1.0, 0.05),
        }
        toggle_keys = {"use_gpu", "smooth", "fxaa",
                       "highlight_points", "auto_iter", "reset-orbit-points"}

        if key == "max_iter":
            val = state["max_iter"]
            if val <= 0:
                val = 1
            log2_val = math.log2(val)
            if is_shift:
                delta = 10 if rtype == "plus" else -10
            elif is_ctrl:
                delta = 1 if rtype == "plus" else -1
            else:
                delta = 1 if rtype == "plus" else -1
            new_log2 = int(math.floor(log2_val + delta))
            new_val = max(1, min(RENDER_ITER_CAP, int(2 ** new_log2)))
            state["max_iter"] = new_val
            return True

        if key == "orbit_max_iter":
            val = state["orbit_max_iter"]
            if val <= 0:
                val = 1
            log2_val = math.log2(val)
            if is_shift:
                delta = 4 if rtype == "plus" else -4
            elif is_ctrl:
                delta = 1 if rtype == "plus" else -1
            else:
                delta = 1 if rtype == "plus" else -1
            new_log2 = int(math.floor(log2_val + delta))
            new_val = max(1, min(ORBIT_ITER_CAP, int(2 ** new_log2)))
            state["orbit_max_iter"] = new_val
            return True

        elif key in rows_meta:
            is_int, val, vmin, vmax, base_step = rows_meta[key]
            multiplier = 10.0 if is_shift else (0.1 if is_ctrl else 1.0)
            step = base_step * multiplier
            grid = base_step if (is_shift or not is_ctrl) else base_step * 0.1
            direction = 1 if rtype == "plus" else -1
            new_val = val + direction * step
            new_val = round(new_val / grid) * grid
            if key in cyclic_keys:
                span = vmax - vmin
                if span > 0:
                    new_val = ((new_val - vmin) % span) + vmin
            else:
                new_val = max(vmin, min(vmax, new_val))
            if is_int:
                new_val = int(new_val)
            state[key] = new_val
            return True
        
        elif key == "show-keybinds":
            state["show_keybinds"] = not state.get("show_keybinds", False)
            return True

        elif key == "keybinds-menu":
            state["show_keybinds"] = True
            return True

        elif key == "points-menu":
            state["show_point_info"] = True
            return True

        elif key == "highlight-points":
            state["highlight_points"] = True
            state["highlight_flash_time"] = pygame.time.get_ticks()
            return True

        elif key == "toggle-point-info":
            state["show_point_info"] = not state.get("show_point_info", False)
            return True

        elif key in toggle_keys:
            if key == "reset-orbit-points":
                state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
                state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
                state["julia_c"] = list(DEFAULT_JULIA_C)
            elif key == "highlight_points":
                state["highlight_points"] = not state.get("highlight_points", False)
                if state["highlight_points"]:
                    state["highlight_flash_time"] = pygame.time.get_ticks()
            else:
                state[key] = not state.get(key, False)
            return True
        
        elif key == "reset-orbit-point":
            state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
            state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
            state["julia_c"] = list(DEFAULT_JULIA_C)
            return True
        
        elif key == "reset-orbit-point-m":
            state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
            return True
        
        elif key == "reset-orbit-point-j":
            state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
            return True
        
        elif key == "reset-julia-c":
            state["julia_c"] = list(DEFAULT_JULIA_C)
            return True
        
        elif key == "reset-all":
            _reset_to_defaults(state)
            return True
        
        elif key in ("cycle-palette", "load-palette-file"):
            palettes = get_all_palettes()
            if not palettes:
                print("[palette] No palettes available to load")
                return False
            _idx = state.get("_palette_file_idx", 0) % len(palettes)
            _apply_palette_by_index(state, _idx)
            state["palette_index"] = _idx
            state["_palette_file_idx"] = (_idx + 1) % len(palettes)
            _verb = "loaded" if key == "load-palette-file" else "cycled to"
            print(f"[palette] {_verb}: {palettes[_idx]['name']}")
            return True
        
        elif key == "toggle-split":
            cur = state.get("split_mode", DEFAULT_SPLIT_MODE)
            _prev_blend = state.get("set_blend", 0.0)
            if cur is None:
                state["_pre_split_blend"] = _prev_blend
                state["julia_viewport"] = list(DEFAULT_JULIA_VIEWPORT)
                state["split_mode"] = "horizontal"
                state["split_orientation"] = "horizontal"
                state["set_blend"] = 0.0
            elif cur == "horizontal":
                state["split_mode"] = "vertical"
                state["split_orientation"] = "vertical"
                state["set_blend"] = 0.0
            else:
                state["split_mode"] = None
                state["set_blend"] = state.get("_pre_split_blend", 0.5)
            return True
        return False

    @staticmethod
    def _compute_zoom_percent(xmin, xmax, ymin, ymax, width, height):
        """Compute zoom level as a percentage relative to the default view.

        Uses log2 scale: zoom_level = log2(default_range / current_range)
        where default_range = 4.0 (width of default Mandelbrot view).
        Percentage = min(100, zoom_level / 40 * 100) where 40 = log2(4.0/1e-10).
        """
        return _zoom_percent(xmin, xmax, ymin, ymax, width, height)

    def draw(self, screen, font, state, keybinds, xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5):
        if not self.active:
            return
        sw, sh = screen.get_size()
        scale = max(0.5, min(1.5, min(sw / 1920, sh / 1080)))
        _n_rows = 25
        _n_toggles = 4
        _n_sections = 4
        _n_zoom = 1                            # Zoom display line
        _n_action_spacer = 2                   # Spacing around action buttons
        _total_items = _n_rows + _n_toggles + _n_sections + _n_zoom + _n_action_spacer + 7
        _row_h_est = max(1, int(28 * scale)) + max(1, int(8 * scale))
        _pad_est = max(1, int(10 * scale))
        _menu_h_est = _total_items * _row_h_est + 2 * _pad_est
        _max_menu_h = sh - 2 * _pad_est
        if _menu_h_est > _max_menu_h:
            scale = min(scale, _max_menu_h / _menu_h_est)
        self.menu_scale = scale

        _cache_key = (
            round(scale, 6), state["max_iter"], int(state["stripe_s"]), int(state["step_s"]),
            state["light_angle"], state["light_azim"], state["light_i"],
            state["k_ambiant"], state["k_diffuse"], state["k_specular"], state["shininess"],
            state["orbit_max_iter"], state.get("point_size", DEFAULT_POINT_SIZE),
            state.get("set_blend", 0.0), state.get("grid_opacity", 0.3),
            state.get("grid_label_opacity", DEFAULT_GRID_LABEL_OPACITY),
            state.get("smooth", True), state.get("fxaa", False),
            state.get("use_gpu", False) and _CUDA_AVAILABLE,
            state.get("highlight_points", DEFAULT_HIGHLIGHT_POINTS),
            state.get("auto_iter", DEFAULT_AUTO_ITER),
            state.get("show_keybinds", False), state.get("show_point_info", False),
            state.get("point_opacity_m", 1.0), state.get("point_opacity_j", 1.0),
            state.get("point_opacity_c", 1.0),
            state.get("orbits_opacity_m", DEFAULT_ORBITS_OPACITY_M),
            state.get("orbits_opacity_j", DEFAULT_ORBITS_OPACITY_J),
            state.get("lines_opacity_m", DEFAULT_LINES_OPACITY_M),
            state.get("lines_opacity_j", DEFAULT_LINES_OPACITY_J),
            state.get("cycle_opacity", 1.0),
            state.get("center_opacity", 1.0),
            state.get("point_size_m", DEFAULT_POINT_SIZE),
            state.get("point_size_j", DEFAULT_POINT_SIZE),
            state.get("point_size_c", DEFAULT_POINT_SIZE),
            state.get("split_mode", DEFAULT_SPLIT_MODE),
            state.get("palette_index", 0),
            state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[0], state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[1],
            state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[0], state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[1],
            state.get("julia_c", DEFAULT_JULIA_C)[0], state.get("julia_c", DEFAULT_JULIA_C)[1],
            tuple(state.get("gradient_stops")) if state.get("gradient_stops") else None,
            state.get("gradient_blend", DEFAULT_GRADIENT_BLEND),
            xmin, xmax, ymin, ymax,
            sw, sh,
        )
        if self._menu_cache is not None and self._menu_cache[0] == _cache_key:
            _surf, _button_rects, _ax, _ay = self._menu_cache[1]
            screen.blit(_surf, (_ax, _ay))
            self.button_rects = _button_rects
            if self.show_keybinds:
                # The menu surface is cached, but the keybinds panel has its own
                # cache and must still be drawn on cache-hit frames.
                self._draw_keybinds(screen, font, 0, 0, keybinds, scale, state)
            return
        s = lambda v: max(1, int(v * scale))
        pad = s(10)
        btn_w = s(36)
        btn_h = s(28)
        label_w = s(160)
        val_w = s(100)
        row_h = btn_h + s(8)
        font_size = max(12, int(24 * scale))
        try:
            scaled_font = pygame.font.SysFont("monospace", font_size)
        except Exception:
            scaled_font = font
        th = scaled_font.get_height()

        sections = [
            ("Parameters", [
                ("max_iter",            "iterations",       state["max_iter"],                  True),
                ("orbit_max_iter",      "orbit_iter",       state["orbit_max_iter"],            True),
            ]),
            ("Lighting", [
                ("light_angle",         "light_angle",      state["light_angle"],               False),
                ("light_azim",          "light_azim",       state["light_azim"],                False),
                ("light_i",             "light_i",          state["light_i"],                   False),
                ("k_ambiant",           "k_amb",            state["k_ambiant"],                 False),
                ("k_diffuse",           "k_diff",           state["k_diffuse"],                False),
                ("k_specular",          "k_spec",           state["k_specular"],               False),
                ("shininess",           "shininess",        state["shininess"],                False),
            ]),
            ("Colors", [
                ("stripe_s",            "stripes",          int(state["stripe_s"]),             True),
                ("step_s",              "steps",            int(state["step_s"]),               True),
                ("set_blend",           "blend",            state.get("set_blend", 0.0),        False),
                ("grid_opacity",        "grid opacity",     state.get("grid_opacity", DEFAULT_GRID_OPACITY), False),
                ("grid_label_opacity",  "grid labels",      state.get("grid_label_opacity", DEFAULT_GRID_LABEL_OPACITY), False),
                ("point_opacity_m",     "m point opacity",  state.get("point_opacity_m", 1.0),  False),
                ("point_opacity_j",     "j point opacity",  state.get("point_opacity_j", 1.0),  False),
                ("point_opacity_c",     "c point opacity",  state.get("point_opacity_c", 1.0),  False),
                ("cycle_opacity",       "cycle pt opacity", state.get("cycle_opacity", DEFAULT_CYCLE_OPACITY), False),
                ("center_opacity",       "center pt opacity", state.get("center_opacity", DEFAULT_CENTER_OPACITY), False),
                ("orbits_opacity_m",    "mb orbits",        state.get("orbits_opacity_m", DEFAULT_ORBITS_OPACITY_M), False),
                ("orbits_opacity_j",    "ju orbits",        state.get("orbits_opacity_j", DEFAULT_ORBITS_OPACITY_J), False),
                ("lines_opacity_m",     "mb lines",         state.get("lines_opacity_m", DEFAULT_LINES_OPACITY_M), False),
                ("lines_opacity_j",     "ju lines",         state.get("lines_opacity_j", DEFAULT_LINES_OPACITY_J), False),
            ]),
            ("Point sizes", [
                ("point_size_m",        "point size m",     state.get("point_size_m", DEFAULT_POINT_SIZE), True),
                ("point_size_j",        "point size j",     state.get("point_size_j", DEFAULT_POINT_SIZE), True),
                ("point_size_c",        "point size c",     state.get("point_size_c", DEFAULT_POINT_SIZE), True),
            ]),
        ]

        toggles = [
            ("use_gpu",                 "GPU",              state["use_gpu"] and _CUDA_AVAILABLE),
            ("smooth",                  "smooth",           state.get("smooth", True)),
            ("fxaa",                    "fxaa",             state.get("fxaa", False)),
            ("auto_iter",               "auto iter",        state.get("auto_iter", DEFAULT_AUTO_ITER)),
        ]

        palettes = get_all_palettes()
        palette_names = [p["name"] for p in palettes]
        palette_idx = state.get("palette_index", 0)
        palette_name = palette_names[palette_idx] if 0 <= palette_idx < len(palette_names) else "none"

        # Consistent fixed menu colors (independent of the active palette).
        menu_bg = _palette_section_color(state, "main menu", 0.0)
        _header_color = _palette_section_color(state, "main menu", 1.0)
        _ct_on  = _palette_section_color(state, "main menu", 0.7)
        _ct_off = _palette_section_color(state, "main menu", 0.15)
        _ct_dim = _palette_section_color(state, "main menu", 0.4)

        action_buttons = [
            ("reset-orbit-point-m", "reset MB point",       _ct_dim),
            ("reset-orbit-point-j", "reset Julia point",    _ct_dim),
            ("reset-julia-c",       "reset c point",        _ct_dim),
            ("reset-all",           "reset all settings",   _ct_dim),
            ("cycle-palette",       f"palette: {palette_name}", _ct_dim),
            ("toggle-split",        "split: h/v/overlay",   _ct_dim),
            ("highlight-points",    "highlight points",     _ct_dim),
        ]

        bottom_toggles = [
            ("toggle-point-info", "points menu",   state.get("show_point_info", False)),
            ("show-keybinds",     "keybinds menu", state.get("show_keybinds", False)),
        ]

        _row_labels = [
            label
            for _, _rows in sections
            for _, label, _, _ in _rows
        ]
        _row_labels.extend(label for _, label, _ in toggles)
        _row_values = [
            str(int(val)) if is_int else f"{val:.2f}"
            for _, _rows in sections
            for _, label, val, is_int in _rows
        ]
        label_w = max(
            label_w,
            max((scaled_font.size(label)[0] for label in _row_labels), default=0) + s(12),
            s(160),
        )
        val_w = max(
            val_w,
            max((scaled_font.size(value)[0] for value in _row_values), default=0) + s(12),
            s(100),
        )
        btn_h = max(btn_h, th)
        row_h = max(row_h, th + s(4))
        act_w = label_w + val_w + 2 * (btn_w + s(5)) + pad
        _max_act_w = max(scaled_font.size(label)[0] for _, label, _ in action_buttons) + 2 * (pad + s(8))
        act_w = max(act_w, _max_act_w)

        menu_w = act_w + 2 * pad
        if  menu_w > sw - pad:
            menu_w = sw - pad
        _n_total_rows = sum(len(_sec_rows) for _, _sec_rows in sections)
        menu_h = (_n_total_rows + len(toggles)
                  + len(action_buttons) + len(bottom_toggles) + len(sections) + 6) * row_h + 2 * pad
        max_menu_h = sh - 2 * pad
        if menu_h > max_menu_h:
            scale = min(scale, max_menu_h / menu_h)
            s = lambda v: max(1, int(v * scale))
            pad = s(10)
            btn_w = s(36)
            font_size = max(12, int(24 * scale))
            try:
                scaled_font = pygame.font.SysFont("monospace", font_size)
            except Exception:
                scaled_font = font
            th = scaled_font.get_height()
            btn_h = max(s(28), th)
            label_w = max(
                s(160),
                max((scaled_font.size(label)[0] for label in _row_labels), default=0) + s(12),
            )
            val_w = max(
                s(100),
                max((scaled_font.size(value)[0] for value in _row_values), default=0) + s(12),
            )
            row_h = max(btn_h + s(8), th + s(4))
            act_w = label_w + val_w + 2 * (btn_w + s(5)) + pad
            _max_act_w = max(scaled_font.size(label)[0] for _, label, _ in action_buttons) + 2 * (pad + s(8))
            act_w = max(act_w, _max_act_w)
            menu_w = act_w + 2 * pad
            if menu_w > sw - pad:
                menu_w = sw - pad
            menu_h = (_n_total_rows + len(toggles)
                      + len(action_buttons) + len(bottom_toggles) + len(sections) + 6) * row_h + 2 * pad
            self.menu_scale = scale
        menu_x = sw - menu_w - pad
        menu_y = pad

        surf = pygame.Surface((menu_w, menu_h), pygame.SRCALPHA)
        surf.fill((*menu_bg, 200))
        pygame.draw.rect(surf, (200, 200, 200), (0, 0, menu_w, menu_h), s(2))

        self.button_rects = {}
        ry = pad

        _zoom_text = f"zoom: {_zoom_percent(xmin, xmax, ymin, ymax, sw, sh):.1f}%"
        _zt = scaled_font.render(_zoom_text, True, _ct_on)
        surf.blit(_zt, (pad, ry + (btn_h - th) // 2))
        ry += row_h

        _header_font_size = max(10, int(18 * scale))
        try:
            _header_font = pygame.font.SysFont("monospace", _header_font_size)
        except Exception:
            _header_font = scaled_font
        _header_h = _header_font.get_height() + s(4)

        for _sec_title, _sec_rows in sections:
            if ry + row_h > menu_h - pad:
                break
            _ht = _header_font.render(f"  {_sec_title}", True, _header_color)
            _hw = _header_font.size(f"  {_sec_title}")[0]
            _hx = pad
            pygame.draw.line(surf, _header_color, (_hx, ry + _header_h - s(2)),
                             (_hx + _hw, ry + _header_h - s(2)), 1)
            surf.blit(_ht, (_hx, ry))
            ry += row_h

            for key, label, val, is_int in _sec_rows:
                if ry + row_h > menu_h - pad:
                    break
                lt = scaled_font.render(label, True, (220, 220, 220))
                surf.blit(lt, (pad, ry + (btn_h - th) // 2))
                if is_int:
                    vt = scaled_font.render(str(int(val)), True, (255, 255, 100))
                else:
                    vt = scaled_font.render(f"{val:.2f}", True, (255, 255, 100))
                surf.blit(vt, (label_w, ry + (btn_h - th) // 2))
                mx = label_w + val_w + pad
                pygame.draw.rect(surf, (60, 60, 60),    (mx, ry, btn_w, btn_h), 0,  s(3))
                pygame.draw.rect(surf, (200, 200, 200), (mx, ry, btn_w, btn_h),     s(1))
                mt = scaled_font.render("-", True, (200, 200, 200))
                mw = scaled_font.size("-")[0]
                surf.blit(mt, (mx + (btn_w - mw) // 2, ry + (btn_h - th) // 2))
                px = mx + btn_w + pad
                pygame.draw.rect(surf, (60, 60, 60),    (px, ry, btn_w, btn_h), 0,  s(3))
                pygame.draw.rect(surf, (200, 200, 200), (px, ry, btn_w, btn_h),     s(1))
                pt = scaled_font.render("+", True, (200, 200, 200))
                pw = scaled_font.size("+")[0]
                surf.blit(pt, (px + (btn_w - pw) // 2, ry + (btn_h - th) // 2))
                self.button_rects[key] = (
                    menu_x + mx, menu_y + ry, btn_w, btn_h,
                    menu_x + px, menu_y + ry, btn_w, btn_h,
                )
                ry += row_h

        def _header(title):
            _ht = _header_font.render(f"  {title}", True, _header_color)
            _hw = _header_font.size(f"  {title}")[0]
            pygame.draw.line(surf, _header_color, (pad, ry + _header_h - s(2)),
                             (pad + _hw, ry + _header_h - s(2)), 1)
            surf.blit(_ht, (pad, ry))

        for _group_title, _group in (("Display", toggles),):
            if not _group:
                continue
            _header(_group_title)
            ry += row_h
            for key, label, val in _group:
                if ry + row_h > menu_h - pad:
                    break
                # Toggle rows: colored ON/OFF buttons (palette-derived variety).
                on_color = _ct_on if val else _ct_off
                pygame.draw.rect(surf, on_color, (pad, ry, act_w, btn_h), 0, s(3))
                pygame.draw.rect(surf, (200, 200, 200), (pad, ry, act_w, btn_h), s(1))
                lt = scaled_font.render(f"{label}: {'ON' if val else 'OFF'}", True, (255, 255, 255))
                surf.blit(lt, (pad + s(8), ry + (btn_h - th) // 2))
                self.button_rects[key] = (menu_x + pad, menu_y + ry, act_w, btn_h)
                ry += row_h

        for btn_key, btn_label, btn_color in action_buttons:
            if ry + row_h > menu_h - pad:
                break
            pygame.draw.rect(surf, btn_color,       (pad, ry, act_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (200, 200, 200), (pad, ry, act_w, btn_h),    s(1))
            bt = scaled_font.render(btn_label, True, (255, 255, 255))
            surf.blit(bt, (pad + s(8), ry + (btn_h - th) // 2))
            self.button_rects[btn_key] = (menu_x + pad, menu_y + ry, act_w, btn_h)
            ry += row_h

        # Points/keybinds menu toggles pinned at the very bottom of the menu.
        for key, label, val in bottom_toggles:
            if ry + row_h > menu_h - pad:
                break
            on_color = _ct_on if val else _ct_off
            pygame.draw.rect(surf, on_color, (pad, ry, act_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (230, 230, 230), (pad, ry, act_w, btn_h), s(2))
            lt = scaled_font.render(f"{label}: {'ON' if val else 'OFF'}", True, (255, 255, 255))
            surf.blit(lt, (pad + s(8), ry + (btn_h - th) // 2))
            self.button_rects[key] = (menu_x + pad, menu_y + ry, act_w, btn_h)
            ry += row_h

        screen.blit(surf, (menu_x, menu_y))
        self._menu_cache = (_cache_key, (surf, dict(self.button_rects), menu_x, menu_y))

        if self.show_keybinds:
            self._draw_keybinds(screen, scaled_font, 0, 0, keybinds, scale, state)

    def _draw_keybinds(self, screen, font, origin_x, origin_y, keybinds, scale, state=None):
        """Draw keybind info panel — vertical, top-left aligned."""
        _kb_bg = _palette_section_color(state, "keybinds menu", 0.0) if state else (0, 0, 0)
        _kb_border = _palette_section_color(state, "keybinds menu", 1.0) if state else (200, 200, 200)
        _kb_text = _palette_section_color(state, "keybinds menu", 0.85) if state else (200, 200, 255)
        _palette_id = state.get("palette_index", 0) if state else -1
        if self._keybind_cache is not None and self._keybind_cache[0] == (scale, screen.get_size(), _palette_id):
            panel_surf, panel_w, panel_h = self._keybind_cache[1]
        else:
            s = lambda v: max(1, int(
                v * scale))
            pad = s(10)
            th = font.get_height()
            row_h = th + s(4)

            items = sorted(
                (k, v) for k, v in keybinds.items()
                if isinstance(v, str) and v.strip()
                and not v.startswith(("scroll_", "mouse"))
            )
            labels = [f"[{kc.upper()}] {action.replace('-', ' ')}" for action, kc in items]
            max_text_w = max((font.size(l)[0] for l in labels), default=0)
            line_w = max(max_text_w + s(20), s(200))
            panel_w = line_w + 2 * pad
            sw, sh = screen.get_size()
            if panel_w > sw - 2 * pad:
                panel_w = sw - 2 * pad
            panel_h = len(items) * row_h + 2 * pad
            max_panel_h = sh - 2 * pad
            if panel_h > max_panel_h:
                panel_h = max_panel_h

            panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel_surf.fill((*_kb_bg, 220))
            pygame.draw.rect(panel_surf, _kb_border, (0, 0, panel_w, panel_h), s(2))

            for i, (action, kc) in enumerate(items):
                ky = pad + i * row_h
                if ky + row_h > panel_h:
                    break
                label = f"[{kc.upper()}] {action.replace('-', ' ')}"
                t = font.render(label, True, _kb_text)
                panel_surf.blit(t, (pad, ky))

            self._keybind_cache = ((scale, sw, sh, _palette_id), (panel_surf, panel_w, panel_h))

        panel_x = origin_x + pad if origin_x else pad
        panel_y = origin_y + pad if origin_y else pad
        # Ensure panel stays within screen bounds and doesn't overlap menu
        sw, sh = screen.get_size()
        if panel_x + panel_w > sw - pad:
            panel_x = max(pad, sw - panel_w - pad)
        if panel_y + panel_h > sh - pad:
            panel_y = max(pad, sh - panel_h - pad)
        screen.blit(panel_surf, (panel_x, panel_y))


def next_pow2(n):
    if n < 1:
        return 2
    if n & (n - 1) == 0:
        return n
    return 1 << n.bit_length()


def prev_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _compute_target_iter(xmin, xmax, ymin, ymax, base_iter=ZOOM_BASE_ITER):
    """Compute target max_iter based on current view range.

    Zoom level is derived from the view's x-range logarithm.  As the user
    zooms in (smaller range), iterations increase logarithmically to
    reveal detail, capped at AUTO_ITER_MAX and floored at AUTO_ITER_MIN.
    """
    view_range = xmax - xmin
    if view_range <= 0:
        return base_iter
    zoom_level = math.log2(4.0 / view_range)
    target = int(base_iter * (2 ** max(0, zoom_level)))
    return max(AUTO_ITER_MIN, min(AUTO_ITER_MAX, target))


def _record_animation(state, width, height, xmin, xmax, ymin, ymax,
                      zoom_to_cx, zoom_to_cy, zoom_factor, n_frames,
                      filename_prefix="anim", output_dir="."):
    """Record a zoom animation to GIF.

    Uses imageio if available; otherwise saves individual frames as PNGs.
    If imageio is not installed, falls back to saving individual PNG frames.
    """
    frames = []
    cur_xmin, cur_xmax = xmin, xmax
    cur_ymin, cur_ymax = ymin, ymax
    max_iter = state.get("max_iter", 256)

    for i in range(n_frames):
        cx = cur_xmin + (cur_xmax - cur_xmin) * zoom_to_cx
        cy = cur_ymax - (cur_ymax - cur_ymin) * zoom_to_cy
        new_w = (cur_xmax - cur_xmin) * zoom_factor
        new_h = (cur_ymax - cur_ymin) * zoom_factor
        cur_xmin = cx - zoom_to_cx * new_w
        cur_xmax = cur_xmin + new_w
        cur_ymax = cy + (1 - zoom_to_cy) * new_h
        cur_ymin = cur_ymax - new_h

        cur_iter = min(max_iter, _compute_target_iter(cur_xmin, cur_xmax, cur_ymin, cur_ymax))
        surface, _ = render_to_surface(
            width, height, cur_xmin, cur_xmax, cur_ymin, cur_ymax, cur_iter, state)
        arr = pygame.surfarray.array3d(surface)
        frames.append(np.transpose(arr, (1, 0, 2)))

    try:
        path = os.path.join(output_dir, f"{filename_prefix}.gif")
        if imageio is None:
            raise ImportError("imageio not installed")
        imageio.mimsave(path, frames, fps=ANIM_FPS)
        print(f"[anim] Saved animation to {path}")
        return path
    except (ImportError, AttributeError):
        if Image is None:
            print("[anim] Neither imageio nor PIL available — cannot save frames")
            return None
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{filename_prefix}_{i:04d}.png")
            img = Image.fromarray(frame)
            img.save(path)
        print(f"[anim] Saved {len(frames)} frames to {output_dir}")
        return None


def _render_exposed_edges(screen, width, height, xmin, xmax, ymin, ymax, state, offset_x, offset_y, params=None):
    """Render only the strips of the screen not covered by the offset-blitted surface."""
    ox = int(offset_x)
    oy = int(offset_y)
    if ox == 0 and oy == 0:
        return

    if params is None:
        params = build_render_params(state, maxiter=state.get("max_iter", 256),
                                     use_julia=state.get("set_blend", 0.0) >= 1.0)

    scale_x = (xmax - xmin) / width if width > 0 else 0
    scale_y = (ymax - ymin) / height if height > 0 else 0

    if ox > 0:
        sx = xmin
        ex = xmin + ox * scale_x
        if sx < ex and ox > 0:
            surf, _ = render_to_surface(ox, height, sx, ex, ymin, ymax, state["max_iter"], state)
            screen.blit(surf, (0, 0))
    elif ox < 0:
        sx = xmax + ox * scale_x
        ex = xmax
        ow = -ox
        if sx < ex and ow > 0:
            surf, _ = render_to_surface(ow, height, sx, ex, ymin, ymax, state["max_iter"], state)
            screen.blit(surf, (width + ox, 0))

    if oy > 0:
        sy = ymax - oy * scale_y
        ey = ymax
        if sy < ey and oy > 0:
            surf, _ = render_to_surface(width, oy, xmin, xmax, sy, ey, state["max_iter"], state)
            screen.blit(surf, (0, 0))
    elif oy < 0:
        sy = ymin
        ey = ymin - oy * scale_y
        oh = -oy
        if sy < ey and oh > 0:
            surf, _ = render_to_surface(width, oh, xmin, xmax, sy, ey, state["max_iter"], state)
            screen.blit(surf, (0, height + oy))


def key_name_to_pygame(name):
    """Convert keybind name to (pygame key constant, modifier mask) tuple.

    Supports formats like 'ctrl+r', 'shift+r', 'ctrl+shift+f', 'r'.
    Returns None for mouse-only names (e.g. 'scroll_up').
    """
    name = name.strip()
    mod_mask = 0
    if "+" in name:
        parts = name.split("+")
        for part in parts[:-1]:
            key = part.strip().lower()
            if key in _MOD_MAP:
                mod_mask |= _MOD_MAP[key]
            else:
                return None
        name = parts[-1].strip()
    lower_name = name.lower()
    if lower_name in ("scroll_up", "scroll_down", "mouse1", "mouse2", "mouse3"):
        kc = None
    elif lower_name == "escape":
        kc = pygame.K_ESCAPE
    elif lower_name == "space":
        kc = pygame.K_SPACE
    elif lower_name in ("enter", "return"):
        kc = pygame.K_RETURN
    elif lower_name == "tab":
        kc = pygame.K_TAB
    elif lower_name == "backspace":
        kc = pygame.K_BACKSPACE
    elif len(name) == 1:
        kc = pygame.key.key_code(name)
    else:
        kc = getattr(pygame, f"K_{lower_name}", None)
    if kc is None:
        return None
    if mod_mask:
        return (kc, mod_mask)
    return kc


def is_shift_held():
    """Check if shift modifier is currently held."""
    return bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)


def step_val(cur, delta, lo, hi, mult=1.0):
    """Step a value by delta*mult, clamped to [lo, hi]."""
    return round(max(lo, min(hi, cur + delta * mult)), 4)


def step_val_cyclic(cur, delta, lo, hi, mult=1.0):
    """Step a value by delta*mult, wrapping cyclically within [lo, hi)."""
    span = hi - lo
    return round(((cur - lo + delta * mult) % span) + lo, 4)


_NUMERIC_STEPS = {
    "stripe-up":   ("stripe_s",    1.0,  0,   101, True,  True),
    "stripe-down": ("stripe_s",   -1.0,  0,   101, True,  True),
    "step-up":     ("step_s",      1.0,  0.0, 100.0, False, True),
    "step-down":   ("step_s",     -1.0,  0.0, 100.0, False, True),
    "light-angle-up":   ("light_angle",  0.1,  0.0, 1.0,  True,  False),
    "light-angle-down": ("light_angle", -0.1,  0.0, 1.0,  True,  False),
    "light-azim-up":    ("light_azim",   0.1,  0.0, 1.0,  True,  False),
    "light-azim-down":  ("light_azim",  -0.1,  0.0, 1.0,  True,  False),
    "light-i-up":       ("light_i",      0.05, 0.0, 1.0,  True,  False),
    "light-i-down":     ("light_i",     -0.05, 0.0, 1.0,  True,  False),
    "k-amb-up":    ("k_ambiant",   0.05, 0.0, 1.0,  True,  False),
    "k-amb-down":  ("k_ambiant",  -0.05, 0.0, 1.0,  True,  False),
    "k-diff-up":   ("k_diffuse",   0.05, 0.0, 1.0,  True,  False),
    "k-diff-down": ("k_diffuse",  -0.05, 0.0, 1.0,  True,  False),
    "k-spec-up":   ("k_specular",  0.05, 0.0, 1.0,  True,  False),
    "k-spec-down": ("k_specular", -0.05, 0.0, 1.0,  True,  False),
    "shininess-up":   ("shininess",  2.0,  0.0, 100.0, False, False),
    "shininess-down": ("shininess", -2.0,  0.0, 100.0, False, False),
    "blend-up":  ("set_blend",  0.05, 0.0, 1.0,  False, False),
    "blend-down": ("set_blend", -0.05, 0.0, 1.0,  False, False),
    "grid-opac-up":    ("grid_opacity",  0.05, 0.0, 1.0,  False, False),
    "grid-opac-down":  ("grid_opacity", -0.05, 0.0, 1.0,  False, False),
}


def _apply_step(state, action, mult):
    """Apply a numeric step action to state using the current shift multiplier."""
    key, delta, lo, hi, cyclic, to_int = _NUMERIC_STEPS[action]
    current = state[key]
    if cyclic:
        new_val = step_val_cyclic(current, delta, lo, hi, mult)
    else:
        new_val = step_val(current, delta, lo, hi, mult)
    if to_int:
        new_val = int(new_val)
    state[key] = new_val


def _reset_to_defaults(state):
    """Reset all visual and interaction settings to their defaults."""
    state["stripe_s"]       = DEFAULT_STRIPE_S
    state["stripe_sig"]     = DEFAULT_STRIPE_SIG
    state["step_s"]         = DEFAULT_STEP_S
    state["light_angle"]    = DEFAULT_LIGHT_ANGLE
    state["light_azim"]     = DEFAULT_LIGHT_AZIM
    state["light_i"]        = DEFAULT_LIGHT_I
    state["k_ambiant"]      = DEFAULT_K_AMBIANT
    state["k_diffuse"]      = DEFAULT_K_DIFFUSE
    state["k_specular"]     = DEFAULT_K_SPECULAR
    state["shininess"]      = DEFAULT_SHININESS
    state["smooth"]     = True
    state["use_gpu"]    = False
    state["split_mode"]             = DEFAULT_SPLIT_MODE
    state["split_orientation"]      = DEFAULT_SPLIT_ORIENT
    state["julia_viewport"]     = list(DEFAULT_JULIA_VIEWPORT)
    state["grid_label_opacity"] = DEFAULT_GRID_LABEL_OPACITY
    state["grid_opacity"]   = DEFAULT_GRID_OPACITY
    state["auto_iter"]          = DEFAULT_AUTO_ITER
    state["c_point_size"]   = DEFAULT_C_POINT_SIZE
    state["orbit_max_iter"]     = DEFAULT_ORBIT_MAX_ITER
    state["orbit_point_size"]   = 3
    state["fxaa"]           = False
    state["palette_index"]      = 0
    _apply_palette_by_index(state, state["palette_index"])
    state["_palette_file_idx"]  = 0
    state["set_blend"]          = DEFAULT_SET_BLEND
    state["orbit_point_m"]  = list(DEFAULT_ORBIT_POINT_M)
    state["orbit_point_j"]  = list(DEFAULT_ORBIT_POINT_J)
    state["julia_c"]        = list(DEFAULT_JULIA_C)
    state["precision_mode"]     = DEFAULT_PRECISION_MODE
    state["precision_bits"]     = DEFAULT_PRECISION_BITS
    state["show_uncertainty"]   = DEFAULT_SHOW_UNCERTAINTY
    state["show_point_info"]    = DEFAULT_SHOW_POINT_INFO
    state["point_output_format"] = DEFAULT_POINT_OUTPUT_FORMAT
    state["cycle_opacity"]      = DEFAULT_CYCLE_OPACITY
    state["cycle_size"]         = DEFAULT_CYCLE_SIZE
    state["center_opacity"]      = DEFAULT_CENTER_OPACITY
    state["center_size"]         = DEFAULT_CENTER_SIZE
    state["highlight_points"]   = DEFAULT_HIGHLIGHT_POINTS
    state["highlight_flash_duration_ms"] = DEFAULT_HIGHLIGHT_FLASH_DURATION_MS
    state["highlight_flash_frequency"] = DEFAULT_HIGHLIGHT_FLASH_FREQUENCY
    state["point_size"]         = DEFAULT_POINT_SIZE
    state["point_size_m"]       = DEFAULT_POINT_SIZE
    state["point_size_j"]       = DEFAULT_POINT_SIZE
    state["point_size_c"]       = DEFAULT_POINT_SIZE
    state["point_opacity_m"]    = 1.0
    state["point_opacity_j"]    = 1.0
    state["point_opacity_c"]    = 1.0
    state["orbits_opacity_m"]   = DEFAULT_ORBITS_OPACITY_M
    state["orbits_opacity_j"]   = DEFAULT_ORBITS_OPACITY_J
    state["lines_opacity_m"]    = DEFAULT_LINES_OPACITY_M
    state["lines_opacity_j"]    = DEFAULT_LINES_OPACITY_J
    state["highlight_flash_time"] = 0
    state["show_keybinds"]      = False


def render_image_cli(args):
    """Headless image rendering from CLI args."""
    import pygame
    pygame.init()

    width = args.size[0] if args.size else 1280
    height = args.size[1] if args.size else 720
    if isinstance(args.size, str):
        size_parts = args.size.split(",")
        if len(size_parts) != 2:
            raise ValueError("--size must be WIDTH,HEIGHT")
        width, height = size_parts
    width, height = int(width), int(height)

    settings = load_settings(SETTINGS_FILE)
    _apply_caps_from_settings(settings)
    max_iter = args.iter if args.iter is not None else \
        get_persistent_setting(settings, "max-iterations", cast=int, default=128)
    precision_mode = _normalize_precision_mode(
        getattr(args, "precision_mode", None) or
        get_persistent_setting(settings, "precision-mode", cast=str,
                               default=DEFAULT_PRECISION_MODE))
    precision_bits = _normalize_precision_bits(
        getattr(args, "precision_bits", None)
        if getattr(args, "precision_bits", None) is not None else
        get_persistent_setting(settings, "precision-bits", cast=int,
                               default=DEFAULT_PRECISION_BITS))
    no_uncertainty = getattr(args, "no_uncertainty", None)
    show_uncertainty = (not no_uncertainty
                        if no_uncertainty is not None else
                        get_persistent_setting(settings, "show-uncertainty",
                                               cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                               default=DEFAULT_SHOW_UNCERTAINTY))

    xmin, xmax, ymin, ymax = (Decimal("-2.5"), Decimal("1.0"),
                              Decimal("-1.5"), Decimal("1.5"))
    if args.center is not None:
        parts = [part.strip() for part in args.center.split(",")]
        if len(parts) != 2:
            raise ValueError("--center must be RE,IM")
        cx, cy = (_as_decimal(parts[0]), _as_decimal(parts[1]))
        zoom = _as_decimal(getattr(args, "zoom", None) or 1)
        if zoom <= 0:
            raise ValueError("--zoom must be greater than zero")
        half_w = (xmax - xmin) / (Decimal(2) * zoom)
        half_h = (ymax - ymin) / (Decimal(2) * zoom)
        xmin, xmax = cx - half_w, cx + half_w
        ymin, ymax = cy - half_h, cy + half_h

    xmin, xmax, ymin, ymax = _fix_aspect_ratio_decimal(
        xmin, xmax, ymin, ymax, width, height)

    state = {
        "max_iter":                 max_iter,
        "use_gpu":                  False,
        "stripe_s":                 DEFAULT_STRIPE_S,
        "stripe_sig":               DEFAULT_STRIPE_SIG,
        "step_s":                   DEFAULT_STEP_S,
        "light_angle":              DEFAULT_LIGHT_ANGLE,
        "light_azim":               DEFAULT_LIGHT_AZIM,
        "light_i":                  DEFAULT_LIGHT_I,
        "k_ambiant":                DEFAULT_K_AMBIANT,
        "k_diffuse":                DEFAULT_K_DIFFUSE,
        "k_specular":               DEFAULT_K_SPECULAR,
        "shininess":                DEFAULT_SHININESS,
        "smooth":                   True,
        "fxaa":                     False,
        "orbit_max_iter":           DEFAULT_ORBIT_MAX_ITER,
        "orbit_point_size":         3,
        "set_blend":                DEFAULT_SET_BLEND,
        "split_mode":               DEFAULT_SPLIT_MODE,
        "split_orientation":        DEFAULT_SPLIT_ORIENT,
        "julia_viewport":           list(DEFAULT_JULIA_VIEWPORT),
        "grid_label_opacity":       DEFAULT_GRID_LABEL_OPACITY,
        "grid_opacity":             DEFAULT_GRID_OPACITY,
        "auto_iter":                DEFAULT_AUTO_ITER,
        "julia_c":                  list(DEFAULT_JULIA_C),
        "c_point_size":             DEFAULT_C_POINT_SIZE,
        "palette_index":            0,
        "_palette_file_idx":        0,
        "precision_mode":           precision_mode,
        "precision_bits":           precision_bits,
        "show_uncertainty":         show_uncertainty,
    }
    _apply_palette_by_index(state, 0)

    view_bounds = (xmin, xmax, ymin, ymax)
    render_params = build_render_params(state, maxiter=max_iter,
                                        view_bounds=view_bounds)
    _t0 = time.perf_counter()
    rgb = _render_single(width, height, xmin, xmax, ymin, ymax, max_iter,
                         render_params)

    render_ms = (time.perf_counter() - _t0) * 1000

    pygame.quit()

    img = Image.fromarray(rgb[::-1], "RGB")
    img.save(args.output)
    print(f"[cli] Saved {width}x{height} image to {args.output} ({render_ms:.0f}ms, iter={max_iter})")


def run_render_mode(settings, cli_iter=None, cli_color=None, cli_gpu=False, cli_no_gpu=False, cli_julia=False, cli_blend=None,
                    cli_precision_mode=None, cli_precision_bits=None, cli_no_uncertainty=None):
    # Honor an on-disk settings file when called with an empty/partial dict so a
    # simulation cannot persist its defaults over the user's real config.
    if not settings:
        settings, settings_load_ok = _load_settings_checked(SETTINGS_FILE)
    else:
        settings_load_ok = True
    _apply_caps_from_settings(settings)
    max_iter = cli_iter if cli_iter is not None else get_persistent_setting(settings, "max-iterations", cast=int, default=128)
    if cli_no_gpu:
        use_gpu = False
    else:
        use_gpu = cli_gpu or (settings.get("use-gpu", "false").lower() == "true")
    precision_mode = _normalize_precision_mode(
        cli_precision_mode if cli_precision_mode is not None else
        get_persistent_setting(settings, "precision-mode", cast=str,
                               default=DEFAULT_PRECISION_MODE))
    precision_bits = _normalize_precision_bits(
        cli_precision_bits if cli_precision_bits is not None else
        get_persistent_setting(settings, "precision-bits", cast=int,
                               default=DEFAULT_PRECISION_BITS))
    show_uncertainty = (not cli_no_uncertainty
                        if cli_no_uncertainty is not None else
                        get_persistent_setting(settings, "show-uncertainty",
                                               cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                               default=DEFAULT_SHOW_UNCERTAINTY))
    init_width = DEFAULT_WINDOW_WIDTH
    init_height = DEFAULT_WINDOW_HEIGHT

    keybinds = load_keybinds(settings, SETTINGS_FILE)

    key_map = {}
    xmin, xmax, ymin, ymax = _parse_main_viewport(settings)
    xmin, xmax, ymin, ymax = fix_aspect_ratio(xmin, xmax, ymin, ymax, init_width, init_height)

    pygame.init()
    screen = pygame.display.set_mode((init_width, init_height), pygame.RESIZABLE)
    pygame.display.set_caption("Mandelbrot/Julia — drag to pan, scroll to zoom, M for info")

    for action, name in keybinds.items():
        if name is None:
            continue
        result = key_name_to_pygame(name)
        if result is not None:
            if isinstance(result, tuple):
                kc, mod_val = result
                key_map.setdefault(kc, []).append((action, mod_val))
            else:
                key_map.setdefault(result, []).append((action, 0))

    screen.fill((30, 30, 30))
    try:
        _font = pygame.font.SysFont("monospace", 24)
        _msg = _font.render("Compiling JIT kernels, please wait...", True,
                            (200, 200, 200))
        _rect = _msg.get_rect(center=(init_width // 2, init_height // 2))
        screen.blit(_msg, _rect)
    except Exception:
        pass
    pygame.display.flip()

    global _dbg_file
    os.makedirs("debug", exist_ok=True)
    _dbg_file = open(
        os.path.join("debug",
                     time.strftime("mandelbrot-debug-%Y-%m-%d_%H-%M-%S.log")),
        "w", buffering=1)
    print(f"[debug] Logging frame timings -> {_dbg_file.name}", file=sys.stderr)
    print(f"[debug] CUDA available: {_CUDA_AVAILABLE}", file=sys.stderr)

    persistent = settings_load_ok and get_persistent_setting(
        settings, "persistent",
        cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
        default=True)

    state = {
        "max_iter": max_iter,
        "use_gpu": use_gpu,
        "stripe_s": get_persistent_setting(settings, "stripe_s", cast=float, default=DEFAULT_STRIPE_S),
        "stripe_sig": get_persistent_setting(settings, "stripe_sig", cast=float, default=DEFAULT_STRIPE_SIG),
        "step_s": get_persistent_setting(settings, "step_s", cast=float, default=DEFAULT_STEP_S),
        "light_angle": get_persistent_setting(settings, "light_angle", cast=float, default=DEFAULT_LIGHT_ANGLE),
        "light_azim": get_persistent_setting(settings, "light_azim", cast=float, default=DEFAULT_LIGHT_AZIM),
        "light_i": get_persistent_setting(settings, "light_i", cast=float, default=DEFAULT_LIGHT_I),
        "k_ambiant": get_persistent_setting(settings, "k_ambiant", cast=float, default=DEFAULT_K_AMBIANT),
        "k_diffuse": get_persistent_setting(settings, "k_diffuse", cast=float, default=DEFAULT_K_DIFFUSE),
        "k_specular": get_persistent_setting(settings, "k_specular", cast=float, default=DEFAULT_K_SPECULAR),
        "shininess": get_persistent_setting(settings, "shininess", cast=float, default=DEFAULT_SHININESS),
        "smooth": get_persistent_setting(settings, "smooth",
                                         cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                         default=True),
         "fxaa": get_persistent_setting(settings, "fxaa",
                                       cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                       default=False),
          "palette_index": get_persistent_setting(settings, "palette-index",
                                                   cast=int, default=0),
         "_palette_file_idx": get_persistent_setting(settings, "palette-file-idx",
                                                     cast=int, default=0),
        "orbit_point_m": [
            get_persistent_setting(settings, "orbit-mx", cast=float, default=DEFAULT_ORBIT_POINT_M[0]),
            get_persistent_setting(settings, "orbit-my", cast=float, default=DEFAULT_ORBIT_POINT_M[1]),
        ],
        "orbit_point_j": [
            get_persistent_setting(settings, "orbit-jx", cast=float, default=DEFAULT_ORBIT_POINT_J[0]),
            get_persistent_setting(settings, "orbit-jy", cast=float, default=DEFAULT_ORBIT_POINT_J[1]),
        ],
        "orbit_max_iter": get_persistent_setting(settings, "max-orbits", cast=int, default=DEFAULT_ORBIT_MAX_ITER),
        "orbit_point_size": get_persistent_setting(settings, "orbit-point-size", cast=int, default=3),
        "set_blend": get_persistent_setting(settings, "set-blend",
                                           cast=float, default=DEFAULT_SET_BLEND),
        "split_mode": get_persistent_setting(settings, "split-mode",
                                             cast=lambda s: str(s).strip().lower() if str(s).strip().lower() != "none" else None,
                                             default=DEFAULT_SPLIT_MODE),
        "split_orientation": get_persistent_setting(settings, "split-orientation",
                                                    cast=lambda s: str(s).strip().lower(),
                                                    default=DEFAULT_SPLIT_ORIENT),
        "julia_viewport": _parse_julia_viewport(settings),
        "grid_label_opacity": get_persistent_setting(settings, "grid-label-opacity",
                                                     cast=float, default=DEFAULT_GRID_LABEL_OPACITY),
        "grid_opacity": get_persistent_setting(settings, "grid-opacity",
                                               cast=float, default=DEFAULT_GRID_OPACITY),
        "auto_iter": get_persistent_setting(settings, "auto-iter",
                                             cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                             default=DEFAULT_AUTO_ITER),
        "precision_mode": precision_mode,
        "precision_bits": precision_bits,
        "show_uncertainty": show_uncertainty,
        "julia_c": [

            get_persistent_setting(settings, "julia-cx", cast=float, default=DEFAULT_JULIA_C[0]),
            get_persistent_setting(settings, "julia-cy", cast=float, default=DEFAULT_JULIA_C[1]),
        ],
        "c_point_size": get_persistent_setting(settings, "c-point-size", cast=int, default=DEFAULT_C_POINT_SIZE),
        "show_point_info": get_persistent_setting(settings, "show-point-info",
                                                     cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                     default=DEFAULT_SHOW_POINT_INFO),
        "point_output_format": get_persistent_setting(settings, "point-output-format",
                                                        cast=str, default=DEFAULT_POINT_OUTPUT_FORMAT),
        "cycle_opacity": get_persistent_setting(settings, "cycle-opacity",
                                                  cast=float, default=DEFAULT_CYCLE_OPACITY),
        "cycle_size": get_persistent_setting(settings, "cycle-size",
                                               cast=int, default=DEFAULT_CYCLE_SIZE),
        "center_opacity": get_persistent_setting(settings, "center-opacity",
                                                  cast=float, default=DEFAULT_CENTER_OPACITY),
        "center_size": get_persistent_setting(settings, "center-size",
                                               cast=int, default=DEFAULT_CENTER_SIZE),
        "highlight_points": get_persistent_setting(settings, "highlight-points",
                                                    cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                    default=DEFAULT_HIGHLIGHT_POINTS),
        "highlight_flash_duration_ms": get_persistent_setting(settings, "highlight-flash-duration-ms",
                                                                cast=int, default=DEFAULT_HIGHLIGHT_FLASH_DURATION_MS),
        "highlight_flash_frequency": get_persistent_setting(settings, "highlight-flash-frequency",
                                                              cast=float, default=DEFAULT_HIGHLIGHT_FLASH_FREQUENCY),
        "point_size": get_persistent_setting(settings, "point-size",
                                             cast=int, default=DEFAULT_POINT_SIZE),
        "point_size_m": get_persistent_setting(settings, "point-size-m",
                                               cast=int, default=DEFAULT_POINT_SIZE),
        "point_size_j": get_persistent_setting(settings, "point-size-j",
                                               cast=int, default=DEFAULT_POINT_SIZE),
        "point_size_c": get_persistent_setting(settings, "point-size-c",
                                               cast=int, default=DEFAULT_POINT_SIZE),
        "point_opacity_m": get_persistent_setting(settings, "point-opacity-m",
                                                  cast=float, default=1.0),
        "point_opacity_j": get_persistent_setting(settings, "point-opacity-j",
                                                  cast=float, default=1.0),
        "point_opacity_c": get_persistent_setting(settings, "point-opacity-c",
                                                  cast=float, default=1.0),
        "orbits_opacity_m": get_persistent_setting(settings, "orbits-opacity-m",
                                                   cast=float, default=DEFAULT_ORBITS_OPACITY_M),
        "orbits_opacity_j": get_persistent_setting(settings, "orbits-opacity-j",
                                                   cast=float, default=DEFAULT_ORBITS_OPACITY_J),
        "lines_opacity_m": get_persistent_setting(settings, "lines-opacity-m",
                                                  cast=float, default=DEFAULT_LINES_OPACITY_M),
        "lines_opacity_j": get_persistent_setting(settings, "lines-opacity-j",
                                                  cast=float, default=DEFAULT_LINES_OPACITY_J),
        "show_keybinds": get_persistent_setting(settings, "show-keybinds",
                                                cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                default=False),
    }
    palettes = get_all_palettes()
    if not (0 <= state.get("palette_index", 0) < len(palettes)):
        state["palette_index"] = 0
    _apply_palette_by_index(state, state.get("palette_index", 0))

    if cli_color is not None:
        _pnames = get_palette_names()
        if cli_color in _pnames:
            state["palette_index"] = _pnames.index(cli_color)
        else:
            _closest = _pnames[0] if _pnames else None
            print(f"[warn] palette '{cli_color}' not found, using '{_closest}'")
            state["palette_index"] = 0 if _pnames else 0
        _apply_palette_by_index(state, state.get("palette_index", 0))
    if cli_blend is not None:
        state["set_blend"] = max(0.0, min(1.0, round(float(cli_blend), 4)))
    elif cli_julia:
        state["set_blend"] = 1.0

    persistence_snapshot = _persistent_snapshot(
        _build_persistent_values(state, xmin, xmax, ymin, ymax, keybinds))
    persistence_last_change = None

    _t0 = time.perf_counter()
    compute_image(256, 256, xmin, xmax, ymin, ymax, 8,
                  build_render_params(state, maxiter=max_iter))
    pygame.event.pump()
    _cpu_warm_ms = (time.perf_counter() - _t0) * 1000
    print(f"[warmup] CPU JIT compiled in {_cpu_warm_ms:.0f} ms")
    _dbg(f"warmup backend=CPU {int(_cpu_warm_ms)}ms")

    if state["use_gpu"] and _CUDA_AVAILABLE:
        _t0 = time.perf_counter()
        try:
            compute_image(256, 256, xmin, xmax, ymin, ymax, 8,
                          build_render_params({**state, "use_gpu": True}, maxiter=max_iter))
            pygame.event.pump()
            _gpu_warm_ms = (time.perf_counter() - _t0) * 1000
            print(f"[warmup] CUDA JIT compiled in {_gpu_warm_ms:.0f} ms")
            _dbg(f"warmup backend=CUDA {int(_gpu_warm_ms)}ms")
        except Exception as e:
            state["use_gpu"] = False
            print(f"[warmup] CUDA warmup failed, falling back to CPU: {e}",
                  file=sys.stderr)
            _dbg(f"warmup CUDA failed: {e}")

    width, height = init_width, init_height
    dragging = False
    orbit_drag = False
    orbit_drag_mode = None
    drag_pane = None
    force_full_render = False
    last_mouse_pos = (0, 0)
    needs_render = True
    needs_overlay_render = False
    surface = None
    interacting = False
    interact_timer = 0
    INTERACT_SETTLE = 100
    AUTO_ITER_DEBOUNCE = 300
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    last_render_xmin = None
    last_render_xmax = None
    last_render_ymin = None
    last_render_ymax = None
    _last_auto_iter_time = 0.0
    _START_TIME = time.time()
    _render_timeout_count = 0

    if _PERF_MONITOR:
        _PERF_MONITOR.start()

    overlay = MenuOverlay()
    point_info = PointInfoOverlay()
    point_info.active = bool(state.get("show_point_info", False))
    overlay.show_keybinds = bool(state.get("show_keybinds", False))
    overlay_font = _font if _font else pygame.font.SysFont("monospace", 18)

    clock = pygame.time.Clock()
    running = True
    frame_num = 0
    _invalid_key_count = 0
    _last_caption = None

    while running:
        _t_frame_start = time.perf_counter()
        _t_render = 0.0
        _used_gpu = False
        still_interacting = False
        needs_overlay_render = False
        events = pygame.event.get()
        _dbg(f"poll_events count={len(events)}")
        for event in events:
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                pass

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    handled = False
                    needs_full = False
                    if point_info.active:
                        handled, needs_full = point_info.handle_click(event.pos, state)
                    if not handled:
                        handled, needs_full = overlay.handle_click(event.pos, state)
                    if handled:
                        if needs_full:
                            _RENDER_CACHE.clear()
                            _SPLIT_PANE_CACHE.clear()
                            force_full_render = True
                            needs_render = True
                        else:
                            needs_overlay_render = True
                        continue
                    _dbg(f"MOUSEBUTTONDOWN {event.pos}")
                    mpx, mpy = event.pos
                    click_targets = []
                    _split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)
                    _pane_bounds = None
                    if _split_mode is not None:
                        _pane_bounds = _compute_pane_bounds(
                            _split_mode, state.get("split_orientation", DEFAULT_SPLIT_ORIENT),
                            xmin, xmax, ymin, ymax, width, height,
                            julia_viewport=state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT))
                        _sync_julia_viewport(state, _pane_bounds)
                        for _p in _pane_bounds:
                            if _p["x"] <= mpx < _p["x"] + _p["w"] and _p["y"] <= mpy < _p["y"] + _p["h"]:
                                drag_pane = _p
                                break
                        else:
                            drag_pane = None
                    else:
                        drag_pane = None

                    set_blend = state.get("set_blend", 0.0)
                    if _pane_bounds is not None:
                        mb_alpha = 1.0
                        set_blend = 0.0
                    else:
                        mb_alpha = 1.0 - set_blend

                    mb_pane = None
                    ju_pane = None
                    if _pane_bounds is not None:
                        mb_pane = _pane_bounds[0] if not _pane_bounds[0]["is_julia"] else _pane_bounds[1]
                        ju_pane = _pane_bounds[1] if mb_pane is _pane_bounds[0] else _pane_bounds[0]

                    mx, my = state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)
                    px, py = _complex_to_screen_pane(mx, my, pane=mb_pane, width=width, height=height,
                                                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    dist_mb = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    click_targets.append((dist_mb, "mandelbrot", mb_alpha))

                    jx, jy = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
                    px, py = _complex_to_screen_pane(jx, jy, pane=ju_pane, width=width, height=height,
                                                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    dist_ju = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    click_targets.append((dist_ju, "julia", set_blend))

                    cx, cy = state.get("julia_c", DEFAULT_JULIA_C)
                    if _pane_bounds is not None:
                        px, py = _complex_to_screen_pane(cx, cy, pane=mb_pane, width=width, height=height,
                                                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                        dist_cj = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    else:
                        px, py = _complex_to_screen_pane(cx, cy, width=width, height=height,
                                                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                        dist_cj = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    click_targets.append((dist_cj, "julia_c", 1.0))

                    found = False
                    for dist, mode, _alpha in sorted(click_targets, key=lambda t: t[0]):
                        if dist < 20:
                            orbit_drag_mode = mode
                            state["orbit_drag_mode"] = mode
                            orbit_drag = True
                            state["orbit_hover"] = True
                            needs_render = True
                            found = True
                            break

                    if found:
                        continue
                    orbit_drag = False
                    dragging = True
                    last_mouse_pos = event.pos
                    interacting = True
                    interact_timer = pygame.time.get_ticks()
                    drag_offset_x = 0.0
                    drag_offset_y = 0.0
                    last_render_xmin = xmin
                    last_render_xmax = xmax
                    last_render_ymin = ymin
                    last_render_ymax = ymax

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    point_info.handle_release()
                    _dbg(f"MOUSEBUTTONUP {event.pos} dragging=False")
                    orbit_drag = False
                    orbit_drag_mode = None
                    drag_pane = None
                    state["orbit_drag_mode"] = None
                    state["orbit_hover"] = False
                    dragging = False
                    drag_offset_x = 0.0
                    drag_offset_y = 0.0
                    needs_render = True

            elif event.type == pygame.MOUSEMOTION:
                if point_info.handle_motion(event.pos):
                    continue
                if orbit_drag:
                    mx, my = event.pos
                    _split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)
                    _pane_bounds = None
                    if _split_mode is not None:
                        _pane_bounds = _compute_pane_bounds(
                            _split_mode, state.get("split_orientation", DEFAULT_SPLIT_ORIENT),
                            xmin, xmax, ymin, ymax, width, height,
                            julia_viewport=state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT))
                        _sync_julia_viewport(state, _pane_bounds)
                    if orbit_drag_mode == "julia" and _pane_bounds is not None:
                        drag_pane = _pane_bounds[0] if _pane_bounds[0]["is_julia"] else _pane_bounds[1]
                    elif orbit_drag_mode in ("mandelbrot", "julia_c") and _pane_bounds is not None:
                        drag_pane = _pane_bounds[0] if not _pane_bounds[0]["is_julia"] else _pane_bounds[1]
                    else:
                        drag_pane = None
                    if drag_pane is not None:
                        bxmin = drag_pane["p_xmin"]
                        bxmax = drag_pane["p_xmax"]
                        bymin = drag_pane["p_ymin"]
                        bymax = drag_pane["p_ymax"]
                        rel_x = mx - drag_pane["x"]
                        rel_y = my - drag_pane["y"]
                        pw = drag_pane["w"]
                        ph = drag_pane["h"]
                        scale_x = (bxmax - bxmin) / pw if pw > 0 else 0
                        scale_y = (bymax - bymin) / ph if ph > 0 else 0
                        new_sx = bxmin + rel_x * scale_x
                        new_sy = bymax - rel_y * scale_y
                    else:
                        scale_x = (xmax - xmin) / width
                        scale_y = (ymax - ymin) / height
                        new_sx = xmin + mx * scale_x
                        new_sy = ymax - my * scale_y
                    if orbit_drag_mode == "julia":
                        state["orbit_point_j"] = [new_sx, new_sy]
                    elif orbit_drag_mode == "julia_c":
                        state["julia_c"] = [new_sx, new_sy]
                    else:
                        state["orbit_point_m"] = [new_sx, new_sy]
                    _dbg(f"ORBIT_DRAG ({orbit_drag_mode}) to ({new_sx:.4f}, {new_sy:.4f})")
                    needs_render = True
                elif dragging:
                    _dbg(f"MOUSEMOTION {event.pos} drag")
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    last_mouse_pos = event.pos

                    if drag_pane is not None:
                        pane_w = drag_pane["w"]
                        pane_h = drag_pane["h"]
                        pb_xmin = drag_pane["p_xmin"]
                        pb_xmax = drag_pane["p_xmax"]
                        pb_ymin = drag_pane["p_ymin"]
                        pb_ymax = drag_pane["p_ymax"]
                        if drag_pane["is_julia"]:
                            scale_x = (pb_xmax - pb_xmin) / pane_w if pane_w > 0 else 0
                            scale_y = (pb_ymax - pb_ymin) / pane_h if pane_h > 0 else 0
                            jv = state["julia_viewport"]
                            jv[0] -= dx * scale_x
                            jv[1] -= dx * scale_x
                            jv[2] += dy * scale_y
                            jv[3] += dy * scale_y
                            drag_offset_x += dx
                            drag_offset_y += dy
                        else:
                            scale_x = (pb_xmax - pb_xmin) / pane_w if pane_w > 0 else 0
                            scale_y = (pb_ymax - pb_ymin) / pane_h if pane_h > 0 else 0
                            xmin -= dx * scale_x
                            xmax -= dx * scale_x
                            ymin += dy * scale_y
                            ymax += dy * scale_y
                            drag_offset_x += dx
                            drag_offset_y += dy
                    else:
                        scale_x = (xmax - xmin) / width
                        scale_y = (ymax - ymin) / height
                        xmin -= dx * scale_x
                        xmax -= dx * scale_x
                        ymin += dy * scale_y
                        ymax += dy * scale_y
                        drag_offset_x += dx
                        drag_offset_y += dy
                    interacting = True
                    interact_timer = pygame.time.get_ticks()
                    needs_render = True
            elif event.type == pygame.MOUSEWHEEL:
                mouse_px, mouse_py = pygame.mouse.get_pos()
                if point_info.active and point_info.contains((mouse_px, mouse_py)):
                    point_info.handle_wheel(event.y)
                    continue
                zoom_factor = 0.8 if event.y > 0 else 1.25

                _split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)
                if _split_mode is not None:
                    _pane_bounds = _compute_pane_bounds(
                        _split_mode, state.get("split_orientation", DEFAULT_SPLIT_ORIENT),
                        xmin, xmax, ymin, ymax, width, height,
                        julia_viewport=state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT))
                    _sync_julia_viewport(state, _pane_bounds)
                    _zoom_pane = None
                    for _p in _pane_bounds:
                        if _p["x"] <= mouse_px < _p["x"] + _p["w"] and _p["y"] <= mouse_py < _p["y"] + _p["h"]:
                            _zoom_pane = _p
                            break
                    if _zoom_pane is not None:
                        p_xmin = _zoom_pane["p_xmin"]
                        p_xmax = _zoom_pane["p_xmax"]
                        p_ymin = _zoom_pane["p_ymin"]
                        p_ymax = _zoom_pane["p_ymax"]
                        p_w, p_h = _zoom_pane["w"], _zoom_pane["h"]
                        p_x0, p_y0 = _zoom_pane["x"], _zoom_pane["y"]
                        rel_x = (mouse_px - p_x0) / p_w if p_w > 0 else 0.5
                        rel_y = (mouse_py - p_y0) / p_h if p_h > 0 else 0.5
                        cx = p_xmin + (p_xmax - p_xmin) * rel_x
                        cy = p_ymax - (p_ymax - p_ymin) * rel_y
                        new_w = (p_xmax - p_xmin) * zoom_factor
                        new_h = (p_ymax - p_ymin) * zoom_factor
                        p_xmin = cx - rel_x * new_w
                        p_xmax = p_xmin + new_w
                        p_ymax = cy + rel_y * new_h
                        p_ymin = p_ymax - new_h
                        if _zoom_pane["is_julia"]:
                            jv = state["julia_viewport"]
                            jv[0], jv[1], jv[2], jv[3] = p_xmin, p_xmax, p_ymin, p_ymax
                        else:
                            xmin = p_xmin
                            xmax = p_xmax
                            ymin = p_ymin
                            ymax = p_ymax
                    _dbg(f"MOUSEWHEEL (split) y={event.y} pos={pygame.mouse.get_pos()}")
                    interacting = True
                    interact_timer = pygame.time.get_ticks()
                    needs_render = True
                else:
                    cx = xmin + (xmax - xmin) * mouse_px / width
                    cy = ymax - (ymax - ymin) * mouse_py / height

                    new_width_range = (xmax - xmin) * zoom_factor
                    new_height_range = (ymax - ymin) * zoom_factor

                    xmin = cx - (mouse_px / width) * new_width_range
                    xmax = xmin + new_width_range
                    ymax = cy + (mouse_py / height) * new_height_range
                    _dbg(f"MOUSEWHEEL y={event.y} pos={pygame.mouse.get_pos()}")
                    ymin = ymax - new_height_range
                interacting = True
                interact_timer = pygame.time.get_ticks()
                needs_render = True

            elif event.type == pygame.KEYDOWN:
                if (point_info.active and event.mod == 0
                        and point_info.handle_key(event.key)):
                    continue
                active_input = overlay
                if active_input.coord_input is not None:
                    if event.key == pygame.K_BACKSPACE:
                        active_input.coord_input_text = active_input.coord_input_text[:-1]
                        needs_overlay_render = True
                        continue
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        changed = active_input._commit_coord_edit(state)
                        active_input.coord_input = None
                        active_input.coord_input_text = ""
                        if hasattr(active_input, "_menu_cache"):
                            active_input._menu_cache = None
                        if changed:
                            _RENDER_CACHE.clear()
                            _SPLIT_PANE_CACHE.clear()
                            force_full_render = True
                            needs_render = True
                        else:
                            needs_overlay_render = True
                        continue
                    elif event.key == pygame.K_ESCAPE:
                        active_input.coord_input = None
                        active_input.coord_input_text = ""
                        if hasattr(active_input, "_menu_cache"):
                            active_input._menu_cache = None
                        needs_overlay_render = True
                        continue
                    elif event.key == pygame.K_SPACE:
                        active_input.coord_input_text += " "
                        needs_overlay_render = True
                        continue
                    elif event.key in (pygame.K_PERIOD, pygame.K_MINUS, pygame.K_PLUS,
                                       pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3,
                                       pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7,
                                       pygame.K_8, pygame.K_9, pygame.K_i):
                        _unicode = event.unicode
                        if _unicode:
                            active_input.coord_input_text += _unicode
                            needs_overlay_render = True
                        continue

                bindings = key_map.get(event.key)
                mods = pygame.key.get_mods()
                result = None
                if bindings:
                    for action, mod_val in bindings:
                        if mod_val != 0 and bool(mods & mod_val):
                            result = (action, mod_val)
                            break
                        if mod_val == 0 and result is None:
                            result = (action, mod_val)
                if result is None:
                    _invalid_key_count += 1
                    if _invalid_key_count >= 5:
                        overlay.active = True
                        overlay.show_keybinds = True
                        overlay._keybind_cache = None
                        overlay._menu_cache = None
                        print("[hint] Press 'm' for menu, 'q' to quit")
                        _invalid_key_count = 0
                    continue
                _invalid_key_count = 0
                action, mod_val = result

                _mult = 10.0 if is_shift_held() else 1.0

                if action in _NUMERIC_STEPS:
                    _apply_step(state, action, _mult)
                    # grid_opacity is overlay-only; no fractal re-render needed
                    if action not in ("grid-opac-up", "grid-opac-down"):
                        _RENDER_CACHE.clear()
                        _SPLIT_PANE_CACHE.clear()
                        force_full_render = True
                        needs_render = True
                    if action in ("grid-opac-up", "grid-opac-down"):
                        needs_overlay_render = True

                elif action == "reset-view":
                    xmin, xmax = -2.5, 1.0
                    ymin, ymax = -1.5, 1.5
                    xmin, xmax, ymin, ymax = fix_aspect_ratio(xmin, xmax, ymin, ymax, width, height)
                    force_full_render = True
                    needs_render = True

                elif action == "iter-up":
                    _shift = is_shift_held()
                    cur = state["max_iter"]
                    _log2 = int(math.log2(cur)) if cur > 0 else 0
                    _new_log2 = _log2 + (10 if _shift else 1)
                    state["max_iter"] = int(2 ** _new_log2)
                    force_full_render = True
                    needs_render = True

                elif action == "iter-down":
                    _shift = is_shift_held()
                    cur = state["max_iter"]
                    _log2 = int(math.log2(cur)) if cur > 0 else 0
                    _new_log2 = max(0, _log2 - (10 if _shift else 1))
                    state["max_iter"] = max(1, int(2 ** _new_log2))
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-gpu":
                    state["use_gpu"] = not (state["use_gpu"] and _CUDA_AVAILABLE)
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    _ORBIT_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "menu":
                    overlay.toggle()
                    overlay._menu_cache = None
                    needs_overlay_render = True

                elif action == "toggle-point-info":
                    state["show_point_info"] = not state.get("show_point_info", False)
                    overlay._menu_cache = None
                    needs_overlay_render = True

                elif action == "show-keybinds":
                    state["show_keybinds"] = not state.get("show_keybinds", False)
                    overlay._keybind_cache = None
                    overlay._menu_cache = None
                    needs_overlay_render = True

                elif action == "keybinds-menu":
                    state["show_keybinds"] = True
                    overlay._keybind_cache = None
                    overlay._menu_cache = None
                    needs_overlay_render = True

                elif action == "points-menu":
                    state["show_point_info"] = True
                    overlay._menu_cache = None
                    needs_overlay_render = True

                elif action == "highlight-points":
                    state["highlight_points"] = True
                    state["highlight_flash_time"] = pygame.time.get_ticks()
                    needs_render = True

                elif action == "toggle-smooth":
                    state["smooth"] = not state["smooth"]
                    if not state["smooth"]:
                        state["step_s"] = 0
                    _RENDER_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "reset-settings":
                    _reset_to_defaults(state)
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-fxaa":
                    state["fxaa"] = not state.get("fxaa", False)
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-highlight-points":
                    state["highlight_points"] = not state.get("highlight_points", False)
                    if state["highlight_points"]:
                        state["highlight_flash_time"] = pygame.time.get_ticks()
                    needs_overlay_render = True

                elif action == "toggle-split":
                    cur = state.get("split_mode", DEFAULT_SPLIT_MODE)
                    _prev_blend = state.get("set_blend", 0.0)
                    if cur is None:
                        state["_pre_split_blend"]   = _prev_blend
                        state["julia_viewport"]     = list(DEFAULT_JULIA_VIEWPORT)
                        state["split_mode"]         = "horizontal"
                        state["split_orientation"]  = "horizontal"
                        state["set_blend"]          = 0.0
                    elif cur == "horizontal":
                        state["split_mode"]         = "vertical"
                        state["split_orientation"]  = "vertical"
                        state["set_blend"]          = 0.0
                    else:
                        state["split_mode"] = None
                        state["set_blend"]  = state.get("_pre_split_blend", 0.5)
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-auto-iter":
                    state["auto_iter"] = not state.get("auto_iter", True)
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "reset-orbit-point":
                    state["orbit_point_m"]  = list(DEFAULT_ORBIT_POINT_M)
                    state["orbit_point_j"]  = list(DEFAULT_ORBIT_POINT_J)
                    state["julia_c"]        = list(DEFAULT_JULIA_C)
                    needs_render = True

                elif action == "reset-orbit-point-m":
                    state["orbit_point_m"]  = list(DEFAULT_ORBIT_POINT_M)
                    needs_overlay_render = True

                elif action == "reset-orbit-point-j":
                    state["orbit_point_j"]  = list(DEFAULT_ORBIT_POINT_J)
                    needs_overlay_render = True

                elif action == "reset-julia-c":
                    state["julia_c"]        = list(DEFAULT_JULIA_C)
                    needs_render = True

                elif action == "swap-orbit-point":
                    state["orbit_point_m"], state["orbit_point_j"] = \
                        list(state["orbit_point_j"]), list(state["orbit_point_m"])
                    needs_overlay_render = True

                if action in ("cycle-palette", "load-palette-file"):
                    _handled = overlay._do_action(action, "click", state)
                    if _handled:
                        overlay._menu_cache = None
                        _RENDER_CACHE.clear()
                        _SPLIT_PANE_CACHE.clear()
                        force_full_render = True
                        needs_render = True
                        continue

                elif action == "animate-zoom":
                    _anim_path = _record_animation(
                        state, width, height, xmin, xmax, ymin, ymax,
                        zoom_to_cx=0.5, zoom_to_cy=0.5,
                        zoom_factor=0.96, n_frames=ANIM_FPS * 3,
                        filename_prefix="mandelbrot_zoom",
                        output_dir=os.path.dirname(os.path.abspath(__file__)))
                    if _anim_path:
                        print(f"[anim] Animation saved: {_anim_path}")

                elif action == "quit":
                    running = False

        point_info.active = bool(state.get("show_point_info", False))
        overlay.show_keybinds = bool(state.get("show_keybinds", False))
        overlay._keybind_cache = None if not overlay.show_keybinds else overlay._keybind_cache

        _sw, _sh = screen.get_size()
        if (_sw, _sh) != (width, height):
            _dbg(f"WINDOW_RESIZE {width}x{height} -> {_sw}x{_sh}")
            width, height = _sw, _sh
            xmin, xmax, ymin, ymax = fix_aspect_ratio(
                xmin, xmax, ymin, ymax, width, height)
            _RENDER_CACHE.clear()
            _SPLIT_PANE_CACHE.clear()
            interacting = True
            interact_timer = pygame.time.get_ticks()
            needs_render = True
            drag_offset_x = 0.0
            drag_offset_y = 0.0

        still_interacting = interacting and (pygame.time.get_ticks() - interact_timer < INTERACT_SETTLE)
        if state.get("auto_iter", DEFAULT_AUTO_ITER) and not still_interacting and not dragging and not force_full_render:
            _now = pygame.time.get_ticks()
            if _now - _last_auto_iter_time >= AUTO_ITER_DEBOUNCE:
                _target = _compute_target_iter(xmin, xmax, ymin, ymax)
                if state.get("split_mode") is not None:
                    jv = state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT)
                    _ju_target = _compute_target_iter(jv[0], jv[1], jv[2], jv[3])
                    _target = max(_target, _ju_target)
                if _target != state["max_iter"]:
                    _old = state["max_iter"]
                    state["max_iter"] = _target
                    _RENDER_CACHE.clear()
                    _SPLIT_PANE_CACHE.clear()
                    force_full_render = True
                    needs_render = True
                    _last_auto_iter_time = _now
                    if _target > _old:
                        print(f"[zoom] iter {_old} -> {_target} (zoom in, auto)")
                    else:
                        print(f"[zoom] iter {_old} -> {_target} (zoom out, auto)")

        if needs_render:
            _split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)
            _in_split_drag = _split_mode is not None and dragging and drag_pane is not None
            do_offset_blit = (still_interacting or dragging) and not force_full_render
            if _in_split_drag:
                do_offset_blit = False

            if do_offset_blit and surface is not None and (abs(drag_offset_x) > 0.1 or abs(drag_offset_y) > 0.1):
                # Fill with edge-clamped surface to avoid gaps
                screen.fill((0, 0, 0, 0))
                _blit_surface_clamped(screen, surface,
                                      int(drag_offset_x), int(drag_offset_y),
                                      width, height)
                _t0 = time.perf_counter()
                _render_exposed_edges(screen, width, height, xmin, xmax,
                                      ymin, ymax, state, drag_offset_x,
                                      drag_offset_y)
                _t_render = time.perf_counter() - _t0
                needs_render = True
            elif do_offset_blit:
                _t0 = time.perf_counter()
                surface, _used_gpu = render_to_surface(
                    width, height, xmin, xmax, ymin, ymax, state["max_iter"], state)
                _t_render = time.perf_counter() - _t0
                last_render_xmin = xmin
                last_render_xmax = xmax
                last_render_ymin = ymin
                last_render_ymax = ymax
                drag_offset_x = 0.0
                drag_offset_y = 0.0
                needs_render = True
            else:
                _t0 = time.perf_counter()
                surface, _used_gpu = render_to_surface(
                    width, height, xmin, xmax, ymin, ymax, state["max_iter"], state)
                _t_render = time.perf_counter() - _t0
                needs_render = False
                force_full_render = False
                drag_offset_x = 0.0
                drag_offset_y = 0.0
                last_render_xmin = xmin
                last_render_xmax = xmax
                last_render_ymin = ymin
                last_render_ymax = ymax
                if _t_render * 1000 > RENDER_TIMEOUT_MS:
                    _old_iter = state["max_iter"]
                    state["max_iter"] = max(AUTO_ITER_MIN, _old_iter // 2)
                    _render_timeout_count += 1
                    if _render_timeout_count <= 3:
                        print(f"[warn] Render timeout: {_t_render*1000:.0f}ms with iter={_old_iter}, "
                              f"reducing to {state['max_iter']}", file=sys.stderr)
                    if _old_iter == state["max_iter"]:
                        print(f"[warn] Render timeout threshold reached, cannot reduce further", file=sys.stderr)
                        _render_timeout_count = 0

        _t_blit = 0.0
        if surface is not None:
            _t_blit0 = time.perf_counter()
            if needs_render and dragging and (abs(drag_offset_x) > 0.1 or abs(drag_offset_y) > 0.1):
                screen.blit(surface, (int(drag_offset_x), int(drag_offset_y)))
            else:
                screen.blit(surface, (0, 0))
            _t_blit = time.perf_counter() - _t_blit0
        else:
            _t_blit = 0.0
        backend = "CUDA" if (state["use_gpu"] and _CUDA_AVAILABLE) else "CPU"
        _split_mode = state.get("split_mode", DEFAULT_SPLIT_MODE)
        _pane_bounds = None
        if _split_mode is not None:
            _pane_bounds = _compute_pane_bounds(
                _split_mode, state.get("split_orientation", DEFAULT_SPLIT_ORIENT),
                xmin, xmax, ymin, ymax, width, height,
                julia_viewport=state.get("julia_viewport", DEFAULT_JULIA_VIEWPORT))
            _sync_julia_viewport(state, _pane_bounds)

        if _pane_bounds is not None:
            for _p in _pane_bounds:
                _draw_grid(screen, state,
                           _p["p_xmin"], _p["p_xmax"],
                           _p["p_ymin"], _p["p_ymax"],
                           _p["w"], _p["h"],
                           origin_x=_p["x"], origin_y=_p["y"],
                           font=overlay_font)
        else:
            _draw_grid(screen, state, xmin, xmax, ymin, ymax, width, height,
                       font=overlay_font)

        _draw_orbit(screen, state, xmin, xmax, ymin, ymax, width, height,
                    pane_bounds=_pane_bounds)
        if overlay.active:
            overlay.draw(screen, overlay_font, state, keybinds, xmin, xmax, ymin, ymax)
        if point_info.active:
            point_info.draw(screen, overlay_font, state, xmin, xmax, ymin, ymax, width, height)
        if _PERF_MONITOR and os.environ.get("MB_DEBUG_PERF"):
            stats = _PERF_MONITOR.get_stats()
            fps = frame_num / max(0.001, time.time() - _START_TIME)
            perf_lines = [
                f"FPS: {fps:.1f}",
                f"Render: {_t_render*1000:.0f}ms",
                f"CPU: {stats['cpu']:.0f}%  Mem: {stats['mem_used']:.1f}/{stats['mem_total']:.1f}GB",
                f"GPU: {stats['gpu_util']:.0f}%  VRAM: {stats['gpu_mem_used']:.0f}/{stats['gpu_mem_total']:.1f}MB",
            ]
            if _RENDER_PROFILER:
                rstats = _RENDER_PROFILER.get_stats()
                perf_lines.append(f"Avg render: {rstats['avg_render_ms']:.0f}ms")
                perf_lines.append(f"Tiles: {rstats['tile_distribution']}")
            _render_perf_text(_font or overlay_font, screen, perf_lines)
        _t_flip0 = time.perf_counter()
        pygame.display.flip()
        _t_flip = time.perf_counter() - _t_flip0
        _caption = f"Mandelbrot/Julia blend={state.get('set_blend', 0.0):.2f} — iter={state['max_iter']} [{backend}]"
        if _caption != _last_caption:
            pygame.display.set_caption(_caption)
            _last_caption = _caption
        _dbg(f"frame={frame_num} abs={time.time():.3f} "
             f"total={(time.perf_counter()-_t_frame_start)*1000:.1f}ms "
             f"render={_t_render*1000:.1f}ms blit={_t_blit*1000:.1f}ms "
             f"flip={_t_flip*1000:.1f}ms "
             f"gpu={_used_gpu} hold={still_interacting} drag={dragging} size={width}x{height} iter={state['max_iter']} events={len(events)}")
        if frame_num % 30 == 0 and os.environ.get("MB_DEBUG_PERF"):
            fps = frame_num / max(0.001, time.time() - _START_TIME)
            perf_str = (f"[perf] frame={frame_num} total={(time.perf_counter() - _t_frame_start)*1000:.1f}ms "
                        f"render={_t_render*1000:.0f}ms blit={_t_blit*1000:.0f}ms "
                        f"flip={_t_flip*1000:.0f}ms gpu={_used_gpu} drag={dragging} "
                        f"events={len(events)} iter={state['max_iter']} fps={fps:.1f}")
            if _RENDER_PROFILER:
                rstats = _RENDER_PROFILER.get_stats()
                perf_str += f" avg_render={rstats['avg_render_ms']:.0f}ms tiles={rstats['tile_distribution']}"
            if _PERF_MONITOR:
                mon = _PERF_MONITOR.get_stats()
                perf_str += f" sys_cpu={mon['cpu']:.0f}% gpu_util={mon['gpu_util']:.0f}%"
            print(perf_str, file=sys.stderr)
        persistence_snapshot, persistence_last_change = _save_persistent_state(
            settings, state, xmin, xmax, ymin, ymax,
            keybinds, persistent, persistence_snapshot, persistence_last_change,
            force=False)
        frame_num += 1
        clock.tick(60)

    persistence_snapshot, persistence_last_change = _save_persistent_state(
        settings, state, xmin, xmax, ymin, ymax,
        keybinds, persistent, persistence_snapshot, persistence_last_change,
        force=True)

    print(f"[debug] use_gpu={state['use_gpu']} cuda_available={_CUDA_AVAILABLE} "
          f"final_size={width}x{height} iter={state['max_iter']} frames={frame_num}",
          file=sys.stderr)

    if _PERF_MONITOR:
        _PERF_MONITOR.stop()

    if _dbg_file is not None:
        _dbg("exit")
        _dbg_file.close()
    pygame.quit()


# Load palettes from YAML file (must come after load_palette_file definition)
_ensure_palettes_loaded()
DEFAULT_PALETTE_INDEX   = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=None, help="Max iterations")
    parser.add_argument("--color", type=str, default=None,
                         help="Color palette name (from palettes.yaml)")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU rendering")
    parser.add_argument("--julia", action="store_true", help="Render Julia set (equivalent to --blend 1.0)")
    parser.add_argument("--blend", type=float, default=None, help="Blend factor: 0=MB only, 1=Julia only")
    parser.add_argument("--precision-mode", type=str, default=None,
                         help="Precision mode: auto, mpfr, float64, or perturbed")
    parser.add_argument("--precision-bits", type=int, default=None,
                         help="MPFR precision in bits (53..4096)")
    parser.add_argument("--no-uncertainty", action="store_true", default=None,
                         help="Do not mark bounded high-precision pixels as uncertain")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help="Output image file (PNG). Implies headless mode.")
    parser.add_argument("--size", "-s", type=str, default=None,
                         help="Image size as WIDTH,HEIGHT (e.g. 1920,1080)")
    parser.add_argument("--center", type=str, default=None,
                         help="Center point as RE,IM (e.g. -0.5,0.0)")
    parser.add_argument("--zoom", type=str, default=None,
                         help="Zoom level (1.0 = default view; accepts decimal notation)")
    parser.add_argument("--palette-file", type=str, default=None,
                         help="Path to custom palette YAML file (default: palettes.yaml)")
    parser.add_argument("--watch", action="store_true",
                         help="Auto-restart when source file changes")
    args = parser.parse_args()

    if args.output:
        if Image is None:
            print("[error] PIL/Pillow required for image export. Install with: pip install Pillow", file=sys.stderr)
            sys.exit(1)
        if args.size:
            try:
                args.size = [int(v) for v in args.size.split(",")]
                if len(args.size) != 2:
                    raise ValueError
            except ValueError:
                print("[error] --size must be WIDTH,HEIGHT (e.g. 1920,1080)", file=sys.stderr)
                sys.exit(1)
        if args.center and args.center.count(",") != 1:
            print("[error] --center must be RE,IM (e.g. -0.5,0.0)", file=sys.stderr)
            sys.exit(1)
        render_image_cli(args)
        sys.exit(0)

    if args.palette_file:
        _loaded = load_palette_file(args.palette_file)
        _all_palettes_cache = _loaded
        print(f"[palette] loaded {len(_loaded)} palettes from {args.palette_file}")
    settings = load_settings(SETTINGS_FILE)

    if args.watch:
        _watch_src = os.path.abspath(__file__)
        _last_mtime = os.path.getmtime(_watch_src)
        while True:
            try:
                run_render_mode(settings, cli_iter=args.iter, cli_color=args.color,
                                cli_gpu=args.gpu, cli_no_gpu=args.no_gpu, cli_julia=args.julia,
                                cli_blend=args.blend, cli_precision_mode=args.precision_mode,
                                cli_precision_bits=args.precision_bits,
                                cli_no_uncertainty=args.no_uncertainty)
            except KeyboardInterrupt:
                break
            _new_mtime = os.path.getmtime(_watch_src)
            if _new_mtime == _last_mtime:
                continue
            print(f"\n[watch] Source file changed, restarting... ({_last_mtime:.3f} -> {_new_mtime:.3f})")
            _last_mtime = _new_mtime
            settings = load_settings(SETTINGS_FILE)
    else:
        run_render_mode(settings, cli_iter=args.iter, cli_color=args.color,
                        cli_gpu=args.gpu, cli_no_gpu=args.no_gpu, cli_julia=args.julia,
                        cli_blend=args.blend, cli_precision_mode=args.precision_mode,
                        cli_precision_bits=args.precision_bits,
                        cli_no_uncertainty=args.no_uncertainty)
