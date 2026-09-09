import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
from numba import njit, prange, float64, int64, config

try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    _CUDA_AVAILABLE = False

import pygame

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
        _dbg_file.write(f"{time.time():.3f} {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        _dbg_file.flush()


DEFAULT_KEYBINDS = {
    "quit": "escape",
    "toggle-info": "m",
    "reset-view": "o",
    "toggle-gpu": "p",
    "toggle-smooth": "u",
    "toggle-fxaa": "j",
    "blend-up": "s",
    "blend-down": "a",
    "toggle-orbits": "space",
    "toggle-orbits-m": "ctrl+space",
    "toggle-orbits-j": "alt+space",
    "toggle-orbit-lines-m": "l",
    "toggle-orbit-lines-j": "shift+l",
    "toggle-c-point": "c",
    "reset-orbit-point": "i",
    "reset-orbit-points": "shift+i",
    "swap-orbit-point": "x",
    "cycle-palette": "tab",
    "reset-settings": "backspace",
    "iter-up": "=",
    "iter-down": "-",
    "hue0-up": "ctrl+r",
    "hue0-down": "ctrl+f",
    "hue1-up": "ctrl+g",
    "hue1-down": "ctrl+v",
    "sat-up": "ctrl+n",
    "sat-down": "ctrl+m",
    "zoom-in": "scroll_up",
    "zoom-out": "scroll_down",
}

_MOD_MAP = {
    "ctrl": pygame.KMOD_CTRL,
    "shift": pygame.KMOD_SHIFT,
    "alt": pygame.KMOD_ALT,
    "meta": pygame.KMOD_GUI,
}

DEFAULT_NCYCLE          = 32
DEFAULT_STRIPE_S        = 0.0
DEFAULT_STRIPE_SIG      = 0.9
DEFAULT_STEP_S          = 0.0
DEFAULT_LIGHT_ANGLE     = 0.46  # 0.125/(2*pi) normalized
DEFAULT_LIGHT_AZIM      = 0.5   # 0.5/(pi/2) normalized
DEFAULT_LIGHT_I         = 0.75
DEFAULT_K_AMBIANT       = 0.2
DEFAULT_K_DIFFUSE       = 0.5
DEFAULT_K_SPECULAR      = 0.5
DEFAULT_SHININESS       = 20.0
DEFAULT_RGB_THETAS      = [0.0, 0.167, 0.95]  # fire (HSV: hue_start, hue_end, sat)
DEFAULT_PHASE           = 0.0
DEFAULT_JULIA_C         = [0.394, 0.338]  # c_J from math.txt
DEFAULT_ORBIT_POINT_M   = [0.018, -0.63]  # s_M from math.txt
DEFAULT_ORBIT_POINT_J   = [0.153, 0.473]  # s_J from math.txt
DEFAULT_SET_BLEND       = 0.0             # 0=MB only, 1=Julia only
DEFAULT_ORBIT_POINT     = DEFAULT_ORBIT_POINT_M  # back-compat
DEFAULT_SHOW_C_POINT    = True
DEFAULT_C_POINT_SIZE    = 5
DEFAULT_C_POINT_COLOR   = [0, 200, 0]     # green
DEFAULT_ORBIT_RGB_THETAS = [0.0, 0.167, 0.95]  # orbit color palette (fire)
DEFAULT_ORBIT_LINE_M    = True            # L_M from math.txt
DEFAULT_ORBIT_LINE_J    = True            # L_J from math.txt
DEFAULT_SHOW_ORBITS_M   = True
DEFAULT_SHOW_ORBITS_J   = True
NCOL                    = 2 ** 12

COLOR_THETAS = [
    [0.000, 0.167, 0.95],  # fire      — red -> orange
    [0.550, 0.650, 0.50],  # deep-sea  — dark blue (low sat)
    [0.500, 0.700, 0.80],  # arctic    — cyan -> blue
    [0.500, 0.625, 0.75],  # ocean     — cyan -> blue
    [0.000, 1.000, 0.85],  # twilight  — full hue spectrum
    [0.833, 0.917, 0.90],  # magenta   — magenta -> pink
    [0.333, 0.833, 0.75],  # aurora    — green -> magenta
    [0.250, 0.417, 0.60],  # forest    — yellow-green -> green
    [0.000, 0.000, 0.00],  # grayscale — pure grayscale
    [0.333, 0.333, 0.00],  # monochrome — grayscale (independent of hue)
    [0.000, 0.125, 0.85],  # sunset    — red -> orange
    [0.083, 0.208, 0.90],  # amber     — orange -> yellow
    [0.500, 0.600, 0.40],  # ice       — cyan -> light blue (low sat)
    [0.300, 0.450, 0.55],  # jade      — green -> yellow-green
    [0.000, 0.083, 0.45],  # copper    — red-orange -> orange (muted)
    [0.667, 0.833, 0.85],  # violet    — blue -> magenta
]
DEFAULT_PALETTE_INDEX = 0
PALETTE_NAMES = [
    "fire",
    "deep-sea",
    "arctic",
    "ocean",
    "twilight",
    "magenta",
    "aurora",
    "forest",
    "grayscale",
    "monochrome",
    "sunset",
    "amber",
    "ice",
    "jade",
    "copper",
    "violet"
]

SETTINGS_FILE = "mandelbrot-testing-playground.yaml"


def load_settings(filename):
    settings = {}
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    key, value = line.split("=", 1)
                    settings[key.strip()] = value.strip()
                except ValueError:
                    print(f"Warning: skipping malformed line: '{line}'")
    except FileNotFoundError:
        print(f"'{filename}' not found — will create it as needed.")
    return settings


def save_settings(filename, settings):
    with open(filename, "w") as f:
        for key, value in settings.items():
            f.write(f"{key} = {value}\n")


def get_keybind(settings, name, default_key):
    """Read a keybind from settings, falling back to default."""
    return settings.get(f"keybind.{name}", default_key).lower()


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


@njit(parallel=True)
def _hsv_to_rgb_vec(hue, sat, val):
    n = len(hue)
    r = np.empty(n)
    g = np.empty(n)
    b = np.empty(n)

    for i in prange(n):
        h6 = hue[i] * 6.0
        sector = int(np.floor(h6)) % 6
        f = h6 - np.floor(h6)

        p = val[i] * (1.0 - sat[i])
        q = val[i] * (1.0 - sat[i] * f)
        t = val[i] * (1.0 - sat[i] * (1.0 - f))

        if sector == 0:
            r[i] = val[i]
            g[i] = t
            b[i] = p
        elif sector == 1:
            r[i] = q
            g[i] = val[i]
            b[i] = p
        elif sector == 2:
            r[i] = p
            g[i] = val[i]
            b[i] = t
        elif sector == 3:
            r[i] = p
            g[i] = q
            b[i] = val[i]
        elif sector == 4:
            r[i] = t
            g[i] = p
            b[i] = val[i]
        else:
            r[i] = val[i]
            g[i] = p
            b[i] = q

    return r, g, b


def make_colortable(rgb_thetas):
    x = np.linspace(0.0, 1.0, NCOL)
    hue_start, hue_end, sat = rgb_thetas[0], rgb_thetas[1], rgb_thetas[2]
    hue = np.mod(hue_start + (hue_end - hue_start) * x, 1.0)
    val = 0.15 + 0.7 * (0.5 + 0.5 * np.sin(2 * math.pi * (x + 0.25)))

    if sat == 0.0:
        r = g = b = val.copy()
    else:
        r, g, b = _hsv_to_rgb_vec(hue, np.full_like(hue, sat), val)

    return np.column_stack((r, g, b)).astype(np.float32)


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
        col_i = round(cniter * ncol) % ncol
        nshader_aa = 0.0
    else:
        sqrt_n = math.sqrt(niter)
        floor_n = float(int(sqrt_n))
        cniter = floor_n / ncycle
        col_i = round(cniter * ncol) % ncol
        frac = sqrt_n - floor_n
        aa_factor = 1.0 / (1.0 + math.exp(-30 * (frac - 0.5)))
        col_i_next = (col_i + 1) % ncol
        nshader_aa = aa_factor

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
    if nshader_aa > 0.0:
        r_aa = overlay(colortable[col_i_next, 0], bright, 1)
        g_aa = overlay(colortable[col_i_next, 1], bright, 1)
        b_aa = overlay(colortable[col_i_next, 2], bright, 1)
        r = r * (1 - nshader_aa) + r_aa * nshader_aa
        g = g * (1 - nshader_aa) + g_aa * nshader_aa
        b = b * (1 - nshader_aa) + b_aa * nshader_aa

    return (_clamp01(r), _clamp01(g), _clamp01(b))


@njit
def smooth_iter(c, maxiter, stripe_s, stripe_sig, use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    esc_radius_2 = 4.0
    if use_julia:
        z = c
        c_const = complex(julia_c_re, julia_c_im)
    else:
        z = 0j
        c_const = c
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a = 0.0
    dz = 1 + 0j
    for n in range(maxiter):
        dz = dz * 2 * z + 1
        z = z * z + c_const
        if stripe:
            stripe_t = (math.sin(stripe_s * math.atan2(z.imag, z.real)) + 1) / 2
        if z.real * z.real + z.imag * z.imag > esc_radius_2:
            modz = abs(z)
            log_ratio = 2 * math.log(modz) / math.log(esc_radius_2)
            smooth_i = 1 - math.log(log_ratio) / math.log(2)
            if stripe:
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig - 1)) +
                            stripe_t * smooth_i * (1 - stripe_sig))
                stripe_a = stripe_a / (1 - stripe_sig ** n *
                                       (1 + smooth_i * (stripe_sig - 1)))
            normal = z / dz
            dem = modz * math.log(modz) / abs(dz) / 2
            return (n + smooth_i, stripe_a, dem, normal)
        if stripe:
            stripe_a = stripe_a * stripe_sig + stripe_t * (1 - stripe_sig)
    return (0.0, 0.0, 0.0, 0j)


@njit(parallel=True)
def compute_set_cpu(creal, cim, maxiter, colortable, ncycle,
                    stripe_s, stripe_sig, step_s, diag, light, smooth=True,
                    use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3), dtype=np.float32)
    for x in prange(xpixels):
        for y in range(ypixels):
            niter, stripe_a, dem, normal = smooth_iter(
                complex(creal[x], cim[y]), maxiter, stripe_s, stripe_sig,
                use_julia, julia_c_re, julia_c_im)
            if niter > 0:
                r, g, b = color_pixel(niter, stripe_a, step_s, dem / diag,
                                      normal, colortable, ncycle, light, smooth)
                mat[y, x, 0] = r
                mat[y, x, 1] = g
                mat[y, x, 2] = b
    return mat


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
            # Inline smooth_iter
            esc_radius_2 = 4.0
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
                    col_i = int(round(cniter * ncol)) % ncol
                    nshader_aa = 0.0
                else:
                    sqrt_n = math.sqrt(niter)
                    floor_n = float(int(sqrt_n))
                    cniter = floor_n / ncycle
                    col_i = int(round(cniter * ncol)) % ncol
                    frac = sqrt_n - floor_n
                    nshader_aa = 1.0 / (1.0 + math.exp(-30 * (frac - 0.5)))
                    col_i_next = (col_i + 1) % ncol

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
                if nshader_aa > 0.0:
                    cr_next = colortable[col_i_next, 0]
                    cg_next = colortable[col_i_next, 1]
                    cb_next = colortable[col_i_next, 2]
                    cr = cr * (1 - nshader_aa) + cr_next * nshader_aa
                    cg = cg * (1 - nshader_aa) + cg_next * nshader_aa
                    cb = cb * (1 - nshader_aa) + cb_next * nshader_aa
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



def build_render_params(state, maxiter=None, use_julia=False):
    """Build compute_image params for a single set mode.

    Pass use_julia=True to get Julia params (renders z^2 + c_J where c_J = state['julia_c']).
    Pass use_julia=False for Mandelbrot params (each pixel is c).
    The blending of both sets is handled by render_to_surface.
    """
    rgb_thetas = list(state["rgb_thetas"])
    phase = state["phase"]
    rgb_with_phase = [rgb_thetas[0] + phase, rgb_thetas[1] + phase, rgb_thetas[2]]
    colortable = make_colortable(np.array(rgb_with_phase, dtype=np.float64))
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
def compute_orbit(sx, sy, max_iter, use_julia=False, julia_c_re=0.0, julia_c_im=0.0):
    """Compute the orbit of point (sx, sy).
    Mandelbrot mode: z0=0, c=(sx,sy) → z_{n+1} = z_n^2 + c
    Julia mode: z0=(sx,sy), c=(julia_c_re, julia_c_im) → z_{n+1} = z_n^2 + c_J
    Returns array of (x, y) points."""
    points = np.zeros((max_iter + 1, 2))
    points[0, 0] = sx
    points[0, 1] = sy
    if use_julia:
        zx, zy = sx, sy
        const_x, const_y = julia_c_re, julia_c_im
    else:
        zx, zy = 0.0, 0.0
        const_x, const_y = sx, sy
    for i in range(max_iter):
        new_zx = zx * zx - zy * zy + const_x
        new_zy = 2 * zx * zy + const_y
        zx, zy = new_zx, new_zy
        points[i + 1, 0] = zx
        points[i + 1, 1] = zy
        if zx * zx + zy * zy > 4.0:
            return points[:i + 2]
    return points


def _mandelbrot_to_screen_all(points, xmin, xmax, ymin, ymax, width, height):
    """Convert all Mandelbrot coordinates to screen coordinates (including off-screen)."""
    dx = width / (xmax - xmin) if (xmax - xmin) != 0 else 1.0
    dy = height / (ymax - ymin) if (ymax - ymin) != 0 else 1.0
    return [((points[i, 0] - xmin) * dx, (ymax - points[i, 1]) * dy)
            for i in range(len(points))]


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


def _orbit_pixel_color(i, smooth, colortable, ncycle):
    """Map orbit iteration index to a color using the same colormap as the graph."""
    ncol = colortable.shape[0] - 1
    if smooth:
        cniter = math.sqrt(i + 1) % ncycle / ncycle
        col_i = round(cniter * ncol) % ncol
        r = colortable[col_i, 0]
        g = colortable[col_i, 1]
        b = colortable[col_i, 2]
    else:
        sqrt_n = math.sqrt(i + 1)
        floor_n = float(int(sqrt_n))
        cniter = floor_n / ncycle
        col_i = round(cniter * ncol) % ncol
        frac = sqrt_n - floor_n
        aa_factor = 1.0 / (1.0 + math.exp(-30 * (frac - 0.5)))
        col_i_next = (col_i + 1) % ncol
        r = colortable[col_i, 0] * (1 - aa_factor) + colortable[col_i_next, 0] * aa_factor
        g = colortable[col_i, 1] * (1 - aa_factor) + colortable[col_i_next, 1] * aa_factor
        b = colortable[col_i, 2] * (1 - aa_factor) + colortable[col_i_next, 2] * aa_factor
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_orbit(screen, state, xmin, xmax, ymin, ymax, width, height):
    """Draw orbits on screen.

    Mandelbrot orbit of s_M (state['orbit_point_m']) with opacity (1 - set_blend).
    Julia orbit of s_J (state['orbit_point_j']) with opacity set_blend.
    Julia constant c_J (state['julia_c']) shown as a colored point when show_c_point is True.
    The draggable endpoint highlight goes to the more visible orbit point.
    """
    set_blend = state.get("set_blend", 0.0)
    mb_alpha = 1.0 - set_blend
    ju_alpha = set_blend

    show_orbits = state.get("show_orbits", True)
    show_mb_orbits = state.get("show_orbits_m", show_orbits)
    show_ju_orbits = state.get("show_orbits_j", show_orbits)
    show_mb_lines = state.get("show_orbit_lines_m", True)
    show_ju_lines = state.get("show_orbit_lines_j", True)
    show_c_point = state.get("show_c_point", True)

    max_iter = min(state.get("orbit_max_iter", 200), 500)
    julia_c = state.get("julia_c", DEFAULT_JULIA_C)
    phase = state.get("phase", 0.0)

    # Separate colortable for orbits
    orbit_rgb = list(state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS))
    orbit_rgb_with_phase = [orbit_rgb[0] + phase, orbit_rgb[1] + phase, orbit_rgb[2]]
    orbit_cache_key = (tuple(orbit_rgb), phase)
    if orbit_cache_key not in _ORBIT_COLOR_CACHE:
        _ORBIT_COLOR_CACHE[orbit_cache_key] = make_colortable(
            np.array(orbit_rgb_with_phase, dtype=np.float64))
    colortable = _ORBIT_COLOR_CACHE[orbit_cache_key]

    ncycle = math.sqrt(max_iter)
    smooth = state.get("smooth", True)
    pt_size = state.get("orbit_point_size", 3)

    orbit_hover = state.get("orbit_hover", False)
    orbit_drag_mode = state.get("orbit_drag_mode", None)

    def _draw_single_orbit(sx, sy, use_julia, c_re, c_im, opacity, show_lines):
        if opacity <= 0.0:
            return
        points = compute_orbit(sx, sy, max_iter, use_julia, c_re, c_im)
        screen_pts = _mandelbrot_to_screen_all(points, xmin, xmax, ymin, ymax, width, height)

        if show_lines:
            for i in range(len(screen_pts) - 1):
                color = _orbit_pixel_color(i, smooth, colortable, ncycle)
                color = (int(color[0] * opacity), int(color[1] * opacity), int(color[2] * opacity))
                clipped = _liang_barsky_clip(
                    screen_pts[i][0], screen_pts[i][1],
                    screen_pts[i + 1][0], screen_pts[i + 1][1],
                    0, 0, width, height)
                if clipped:
                    pygame.draw.line(screen, color, (clipped[0], clipped[1]), (clipped[2], clipped[3]), 1)

        for i in range(len(screen_pts)):
            if 0 <= screen_pts[i][0] <= width and 0 <= screen_pts[i][1] <= height:
                color = _orbit_pixel_color(i, smooth, colortable, ncycle)
                color = (int(color[0] * opacity), int(color[1] * opacity), int(color[2] * opacity))
                radius = max(1, int(pt_size * (0.5 + 0.5 * i / len(screen_pts))))
                pygame.draw.circle(screen, color, (int(screen_pts[i][0]), int(screen_pts[i][1])), radius)

    # Draw Mandelbrot orbit (s_M) with opacity = mb_alpha
    orbit_pt_m = state.get("orbit_point_m", state.get("orbit_point", DEFAULT_ORBIT_POINT_M))
    if show_mb_orbits and mb_alpha > 0.0:
        _draw_single_orbit(orbit_pt_m[0], orbit_pt_m[1], False,
                           state.get("julia_c", DEFAULT_JULIA_C)[0],
                           state.get("julia_c", DEFAULT_JULIA_C)[1], mb_alpha, show_mb_lines)

    # Draw Julia orbit (s_J) with opacity = ju_alpha
    orbit_pt_j = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
    if show_ju_orbits and ju_alpha > 0.0:
        _draw_single_orbit(orbit_pt_j[0], orbit_pt_j[1], True,
                           julia_c[0], julia_c[1], ju_alpha, show_ju_lines)

    # Draw c_J (Julia constant) as a colored point
    if show_c_point:
        cx, cy = julia_c
        cx_px = int((cx - xmin) / (xmax - xmin) * width) if (xmax - xmin) > 0 else width // 2
        cy_px = int((ymax - cy) / (ymax - ymin) * height) if (ymax - ymin) > 0 else height // 2
        c_size = state.get("c_point_size", DEFAULT_C_POINT_SIZE)
        c_color = state.get("c_point_color", DEFAULT_C_POINT_COLOR)
        if 0 <= cx_px <= width and 0 <= cy_px <= height:
            pygame.draw.circle(screen, tuple(int(c) for c in c_color), (cx_px, cy_px), max(3, c_size))
            if orbit_drag_mode == "julia_c":
                pygame.draw.circle(screen, (180, 180, 180), (cx_px, cy_px), max(4, c_size + 1), 1)

    # Draw draggable endpoint highlight for the primary visible orbit point
    mb_point_visible = show_mb_orbits and mb_alpha > 0.0
    ju_point_visible = show_ju_orbits and ju_alpha > 0.0
    if mb_point_visible or ju_point_visible:
        if mb_point_visible and (not ju_point_visible or mb_alpha >= ju_alpha):
            px_sx, px_sy = orbit_pt_m
        else:
            px_sx, px_sy = orbit_pt_j

        px = int((px_sx - xmin) / (xmax - xmin) * width) if (xmax - xmin) > 0 else width // 2
        py = int((ymax - px_sy) / (ymax - ymin) * height) if (ymax - ymin) > 0 else height // 2
        bright = int(255 * max(mb_alpha, ju_alpha))
        pygame.draw.circle(screen, (bright, bright, 0), (px, py), 4)
        if orbit_hover:
            outline = int(255 * max(mb_alpha, ju_alpha))
            pygame.draw.circle(screen, (outline, outline, outline), (px, py), 5, 1)


def compute_image(width, height, xmin, xmax, ymin, ymax, maxiter, params,
                  mb_params=None, ju_alpha=0.0):
    """Render a single set or blend two sets.

    If mb_params is None or ju_alpha is 0.0: render single set from params.
    If mb_params is provided and ju_alpha > 0: render Julia from params (ju_alpha),
    blend with Mandelbrot from mb_params (1 - ju_alpha).
    """
    if mb_params is None or ju_alpha <= 0.0:
        return _render_single(width, height, xmin, xmax, ymin, ymax, maxiter, params)

    # Blend mode: render both sets
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
    diag = math.sqrt((xmin - xmax) ** 2 + (ymin - ymax) ** 2)
    smooth = params.get("smooth", True)
    use_julia = params.get("use_julia", False)
    julia_c_re = params.get("julia_c_re", 0.0)
    julia_c_im = params.get("julia_c_im", 0.0)

    if params["use_gpu"] and cuda is not None:
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
                int64(0), int64(0), float64(xmin), float64(xmax),
                float64(ymin), float64(ymax), int64(maxiter), colortable_d,
                float64(ncycle), float64(stripe_s), float64(stripe_sig),
                float64(step_s), float64(diag), light_d, int64(smooth),
                int64(use_julia), float64(julia_c_re), float64(julia_c_im))
            cuda.synchronize()

            if _RENDER_PROFILER:
                _RENDER_PROFILER.record_render(0, 1, 0)

            result = mat.copy_to_host()
            return _post_process(result, params.get("fxaa", False))
        except Exception as e:
            if os.environ.get("MB_DEBUG_PERF"):
                print(f"[gpu] falling back to CPU: {e}", file=sys.stderr)

    mat = compute_set_cpu(np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height),
                          maxiter, colortable, ncycle,
                          stripe_s, stripe_sig, step_s, diag, light,
                          smooth, use_julia, julia_c_re, julia_c_im)
    if _RENDER_PROFILER:
        _RENDER_PROFILER.record_render(0, 0, 1)
    return _post_process(mat, params.get("fxaa", False))




def fix_aspect_ratio(x_min, x_max, y_min, y_max, width, height):
    x_range = x_max - x_min
    y_range = y_max - y_min
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
    max_iter = get_persistent_setting(settings, "iteration-max", cast=int, default=100)
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
_ORBIT_COLOR_CACHE: dict = {}
_PERF_STATS = {"render_count": 0, "total_render_ms": 0.0}


def render_to_surface(width, height, xmin, xmax, ymin, ymax, max_iter, state):
    """Render the set(s) to a pygame Surface.

    Uses set_blend from state to blend Mandelbrot (alpha = 1 - blend) and
    Julia (alpha = blend) sets. Skips rendering a set when its alpha is 0.
    """
    set_blend = max(0.0, min(1.0, state.get("set_blend", 0.0)))
    mb_alpha = 1.0 - set_blend

    _RENDER_CACHE.clear()
    used_gpu = False

    mb_rgb = None
    if mb_alpha > 0.0:
        mb_key = (tuple(state["rgb_thetas"]), state["phase"],
                  state["use_gpu"] and _CUDA_AVAILABLE,
                  max_iter, False, tuple(state.get("julia_c", DEFAULT_JULIA_C)),
                  state.get("smooth", True), state.get("fxaa", False),
                  state.get("stripe_s", 0.0), state.get("stripe_sig", 0.9),
                  state.get("step_s", 0.0), state.get("light_angle", DEFAULT_LIGHT_ANGLE),
                  state.get("light_azim", DEFAULT_LIGHT_AZIM), state.get("light_i", DEFAULT_LIGHT_I),
                  state.get("k_ambiant", DEFAULT_K_AMBIANT), state.get("k_diffuse", DEFAULT_K_DIFFUSE),
                  state.get("k_specular", DEFAULT_K_SPECULAR), state.get("shininess", DEFAULT_SHININESS))
        mb_params = _RENDER_CACHE.get(mb_key)
        if mb_params is None:
            mb_params = build_render_params(state, maxiter=max_iter, use_julia=False)
            _RENDER_CACHE[mb_key] = mb_params
        mb_rgb = _render_single(width, height, xmin, xmax, ymin, ymax, max_iter, mb_params)
        used_gpu = mb_params["use_gpu"]

    if set_blend > 0.0:
        ju_key = (tuple(state["rgb_thetas"]), state["phase"],
                  state["use_gpu"] and _CUDA_AVAILABLE,
                  max_iter, True, tuple(state.get("julia_c", DEFAULT_JULIA_C)),
                  state.get("smooth", True), state.get("fxaa", False),
                  state.get("stripe_s", 0.0), state.get("stripe_sig", 0.9),
                  state.get("step_s", 0.0), state.get("light_angle", DEFAULT_LIGHT_ANGLE),
                  state.get("light_azim", DEFAULT_LIGHT_AZIM), state.get("light_i", DEFAULT_LIGHT_I),
                  state.get("k_ambiant", DEFAULT_K_AMBIANT), state.get("k_diffuse", DEFAULT_K_DIFFUSE),
                  state.get("k_specular", DEFAULT_K_SPECULAR), state.get("shininess", DEFAULT_SHININESS))
        ju_params = _RENDER_CACHE.get(ju_key)
        if ju_params is None:
            ju_params = build_render_params(state, maxiter=max_iter, use_julia=True)
            _RENDER_CACHE[ju_key] = ju_params
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

    def toggle(self):
        self.active = not self.active

    def handle_click(self, pos, state):
        """Handle a mouse click on the menu. Returns True if click was on a button."""
        if not self.active or not self.button_rects:
            return False
        x, y = pos
        for key, rect_info in self.button_rects.items():
            rects = self._unpack_rects(rect_info, key)
            for r, rtype in rects:
                if r.collidepoint(x, y):
                    if key == "show-keybinds":
                        self.show_keybinds = not self.show_keybinds
                        return True
                    self._do_action(key, rtype, state)
                    return True
        return False

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
        cyclic_keys = {"phase", "light_angle", "light_azim", "light_i",
                       "k_ambiant", "k_diffuse", "k_specular",
                       "hue_0", "hue_1", "sat"}
        rows_meta = {
            "max_iter":         (True,  state["max_iter"],         1,      9999, 1.0),
            "stripe_s":         (True,  int(state["stripe_s"]),     0,      100,  1.0),
            "step_s":           (True,  int(state["step_s"]),       0,      100,  1.0),
            "orbit_max_iter":   (True,  state["orbit_max_iter"],    1,      500,  10.0),
            "orbit_point_size": (True,  state.get("orbit_point_size", 3), 1,   10,   1.0),
            "c_point_size":     (True,  state.get("c_point_size", 5), 1,   20,   1.0),
            "shininess":        (False, float(state["shininess"]), 1.0,   100.0, 1.0),
            "phase":            (False, state["phase"],             0.0,   1.0,  0.01),
            "light_angle":      (False, state["light_angle"],       0.0,   1.0,  0.1),
            "light_azim":       (False, state["light_azim"],        0.0,   1.0,  0.1),
            "light_i":          (False, state["light_i"],           0.0,   1.0,  0.01),
            "k_ambiant":        (False, state["k_ambiant"],         0.0,   1.0,  0.01),
            "k_diffuse":        (False, state["k_diffuse"],         0.0,   1.0,  0.01),
            "k_specular":       (False, state["k_specular"],        0.0,   1.0,  0.01),
            "hue_0":            (False, state["rgb_thetas"][0],     0.0,   1.0,  0.01),
            "hue_1":            (False, state["rgb_thetas"][1],     0.0,   1.0,  0.01),
            "sat":              (False, state["rgb_thetas"][2],     0.0,   1.0,  0.01),
            "orbit_hue_0":      (False, state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[0], 0.0, 1.0, 0.01),
            "orbit_hue_1":      (False, state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[1], 0.0, 1.0, 0.01),
            "orbit_sat":        (False, state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[2], 0.0, 1.0, 0.01),
            "c_color_r":        (False, state.get("c_point_color", DEFAULT_C_POINT_COLOR)[0], 0.0, 1.0, 0.02),
            "c_color_g":        (False, state.get("c_point_color", DEFAULT_C_POINT_COLOR)[1], 0.0, 1.0, 0.02),
            "c_color_b":        (False, state.get("c_point_color", DEFAULT_C_POINT_COLOR)[2], 0.0, 1.0, 0.02),
            "set_blend":        (False, round(state.get("set_blend", 0.0), 4), 0.0, 1.0, 0.05),
        }
        toggle_keys = {"use_gpu", "smooth", "fxaa", "show_orbits", "show_orbits_m", "show_orbits_j",
                       "show_orbit_lines_m", "show_orbit_lines_j", "show_c_point", "reset-orbit-points"}

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
            new_val = max(1, min(2**15, int(2 ** new_log2)))
            state["max_iter"] = new_val
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
            if key in ("hue_0", "hue_1", "sat"):
                idx = {"hue_0": 0, "hue_1": 1, "sat": 2}[key]
                state["rgb_thetas"][idx] = new_val
                state["palette_index"] = -1
            elif key in ("orbit_hue_0", "orbit_hue_1", "orbit_sat"):
                idx = {"orbit_hue_0": 0, "orbit_hue_1": 1, "orbit_sat": 2}[key]
                state["orbit_rgb_thetas"][idx] = new_val
            elif key in ("c_color_r", "c_color_g", "c_color_b"):
                idx = {"c_color_r": 0, "c_color_g": 1, "c_color_b": 2}[key]
                state["c_point_color"][idx] = new_val
            else:
                state[key] = new_val
            return True
        elif key in toggle_keys:
            if key == "show_orbits_m":
                state["show_orbits_m"] = not state.get("show_orbits_m", True)
                state["show_orbits"] = state.get("show_orbits_m", True) or state.get("show_orbits_j", True)
            elif key == "show_orbits_j":
                state["show_orbits_j"] = not state.get("show_orbits_j", True)
                state["show_orbits"] = state.get("show_orbits_m", True) or state.get("show_orbits_j", True)
            elif key == "show_orbits":
                state["show_orbits"] = not state.get("show_orbits", True)
            elif key == "reset-orbit-points":
                state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
                state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
                state["orbit_point"] = list(DEFAULT_ORBIT_POINT_M)
                state["julia_c"] = list(DEFAULT_JULIA_C)
            else:
                state[key] = not state.get(key, False)
            return True
        elif key == "reset-orbit-point":
            state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
            state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
            state["orbit_point"] = list(DEFAULT_ORBIT_POINT_M)  # back-compat
            state["julia_c"] = list(DEFAULT_JULIA_C)
            return True
        elif key == "reset-colors":
            state["rgb_thetas"] = list(DEFAULT_RGB_THETAS)
            state["phase"] = DEFAULT_PHASE
            return True
        elif key == "reset-all":
            _reset_to_defaults(state)
            return True
        elif key == "cycle-palette":
            state["palette_index"] = (state.get("palette_index", 0) + 1) % len(COLOR_THETAS)
            t = COLOR_THETAS[state["palette_index"]]
            state["rgb_thetas"] = [t[0], t[1], t[2]]
            return True
        return False

    def draw(self, screen, font, state, keybinds):
        if not self.active:
            return
        sw, sh = screen.get_size()
        scale = max(0.5, min(1.5, min(sw / 1920, sh / 1080)))
        self.menu_scale = scale
        s = lambda v: max(1, int(v * scale))
        pad = s(10)
        btn_w = s(36)
        btn_h = s(28)
        label_w = s(160)
        val_w = s(100)
        row_h = btn_h + s(8)
        act_w = label_w + val_w + 2 * (btn_w + s(5)) + pad  # full-width action button

        font_size = max(12, int(24 * scale))
        try:
            scaled_font = pygame.font.SysFont("monospace", font_size)
        except Exception:
            scaled_font = font
        th = scaled_font.get_height()

        rows = [
            ("max_iter",            "iterations",       state["max_iter"],                  True),
            ("stripe_s",            "stripe_s",     int(state["stripe_s"]),                 True),
            ("step_s",              "step_s",       int(state["step_s"]),                   True),
            ("phase",               "phase",            state["phase"],                     False),
            ("light_angle",         "light_angle",      state["light_angle"],               False),
            ("light_azim",          "light_azim",       state["light_azim"],                False),
            ("light_i",             "light_i",          state["light_i"],                   False),
            ("k_ambiant",           "k_amb",            state["k_ambiant"],                 False),
            ("k_diffuse",           "k_diff",           state["k_diffuse"],                 False),
            ("k_specular",          "k_spec",           state["k_specular"],                False),
            ("shininess",           "shininess",        state["shininess"],                 False),
            ("hue_0",               "hue0",             state["rgb_thetas"][0],             False),
            ("hue_1",               "hue1",             state["rgb_thetas"][1],             False),
            ("sat",                 "sat",              state["rgb_thetas"][2],             False),
            ("orbit_max_iter",      "orbit_iter",       state["orbit_max_iter"],            True),
            ("orbit_point_size",    "orbit_size",       state.get("orbit_point_size", 3),   True),
            ("c_point_size",        "c_pt_size",        state.get("c_point_size", 5),        True),
            ("orbit_hue_0",         "orbit_h0",         state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[0], False),
            ("orbit_hue_1",         "orbit_h1",         state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[1], False),
            ("orbit_sat",           "orbit_sat",        state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[2], False),
            ("c_color_r",           "c_r",              state.get("c_point_color", DEFAULT_C_POINT_COLOR)[0], False),
            ("c_color_g",           "c_g",              state.get("c_point_color", DEFAULT_C_POINT_COLOR)[1], False),
            ("c_color_b",           "c_b",              state.get("c_point_color", DEFAULT_C_POINT_COLOR)[2], False),
            ("set_blend",           "blend",            state.get("set_blend", 0.0),        False),
        ]

        toggles = [
            ("use_gpu", "GPU", state["use_gpu"] and _CUDA_AVAILABLE),
            ("smooth", "smooth", state.get("smooth", True)),
            ("fxaa", "fxaa", state.get("fxaa", False)),
            ("show_orbits_m", "mb orbits", state.get("show_orbits_m", state.get("show_orbits", True))),
            ("show_orbits_j", "ju orbits", state.get("show_orbits_j", state.get("show_orbits", True))),
            ("show_orbit_lines_m", "mb lines", state.get("show_orbit_lines_m", True)),
            ("show_orbit_lines_j", "ju lines", state.get("show_orbit_lines_j", True)),
            ("show_c_point", "c point", state.get("show_c_point", True)),
        ]

        palette_idx = state.get("palette_index", 0)
        palette_name = PALETTE_NAMES[palette_idx] if 0 <= palette_idx < len(PALETTE_NAMES) else "custom"

        menu_w = label_w + val_w + 2 * (btn_w + s(5)) + 4 * pad
        menu_h = (len(rows) + len(toggles) + 6) * row_h + 2 * pad
        menu_x = sw - menu_w - pad
        menu_y = pad

        surf = pygame.Surface((menu_w, menu_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 200))
        pygame.draw.rect(surf, (200, 200, 200), (0, 0, menu_w, menu_h), s(2))

        self.button_rects = {}
        ry = pad

        for key, label, val, is_int in rows:
            lt = scaled_font.render(label, True, (220, 220, 220))
            surf.blit(lt, (pad, ry + (btn_h - th) // 2))
            if is_int:
                vt = scaled_font.render(str(int(val)), True, (255, 255, 100))
            else:
                vt = scaled_font.render(f"{val:.2f}", True, (255, 255, 100))
            surf.blit(vt, (label_w, ry + (btn_h - th) // 2))
            mx = label_w + val_w + pad
            pygame.draw.rect(surf, (60, 60, 60), (mx, ry, btn_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (200, 200, 200), (mx, ry, btn_w, btn_h), s(1))
            mt = scaled_font.render("-", True, (200, 200, 200))
            mw = scaled_font.size("-")[0]
            surf.blit(mt, (mx + (btn_w - mw) // 2, ry + (btn_h - th) // 2))
            px = mx + btn_w + pad
            pygame.draw.rect(surf, (60, 60, 60), (px, ry, btn_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (200, 200, 200), (px, ry, btn_w, btn_h), s(1))
            pt = scaled_font.render("+", True, (200, 200, 200))
            pw = scaled_font.size("+")[0]
            surf.blit(pt, (px + (btn_w - pw) // 2, ry + (btn_h - th) // 2))
            self.button_rects[key] = (
                menu_x + mx, menu_y + ry, btn_w, btn_h,
                menu_x + px, menu_y + ry, btn_w, btn_h,
            )
            ry += row_h

        for key, label, val in toggles:
            on_color = (80, 180, 80) if val else (180, 80, 80)
            pygame.draw.rect(surf, on_color, (pad, ry, act_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (200, 200, 200), (pad, ry, act_w, btn_h), s(1))
            lt = scaled_font.render(f"{label}: {'ON' if val else 'OFF'}", True, (255, 255, 255))
            surf.blit(lt, (pad + s(8), ry + (btn_h - th) // 2))
            self.button_rects[key] = (menu_x + pad, menu_y + ry, act_w, btn_h)
            ry += row_h

        # Full-width action buttons
        for btn_key, btn_label, btn_color in [
            ("reset-orbit-point", "reset orbit point [I]", (60, 60, 60)),
            ("reset-colors", "reset colors (RGB+phase)", (80, 60, 60)),
            ("reset-all", "reset all settings [BS]", (80, 60, 60)),
            ("reset-orbit-points", "reset points (MB+Julia+c_J)", (60, 60, 70)),
            ("cycle-palette", f"palette: {palette_name} [TAB]", (60, 60, 60)),
            ("show-keybinds", "show keybinds [K]", (50, 50, 70)),
        ]:
            pygame.draw.rect(surf, btn_color, (pad, ry, act_w, btn_h), 0, s(3))
            pygame.draw.rect(surf, (200, 200, 200), (pad, ry, act_w, btn_h), s(1))
            bt = scaled_font.render(btn_label, True, (255, 255, 255))
            surf.blit(bt, (pad + s(8), ry + (btn_h - th) // 2))
            self.button_rects[btn_key] = (menu_x + pad, menu_y + ry, act_w, btn_h)
            ry += row_h

        screen.blit(surf, (menu_x, menu_y))

        if self.show_keybinds:
            self._draw_keybinds(screen, scaled_font, 0, 0, keybinds, scale)

    def _draw_keybinds(self, screen, font, origin_x, origin_y, keybinds, scale):
        """Draw keybind info panel — vertical, top-left aligned."""
        s = lambda v: max(1, int(v * scale))
        pad = s(10)
        th = font.get_height()
        row_h = th + s(4)
        line_w = s(280)

        items = sorted(keybinds.items())
        panel_w = line_w + 2 * pad
        panel_h = len(items) * row_h + 2 * pad
        panel_x = pad
        panel_y = pad

        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 220))
        pygame.draw.rect(surf, (200, 200, 200), (0, 0, panel_w, panel_h), s(2))

        for i, (action, kc) in enumerate(items):
            ky = pad + i * row_h
            label = f"[{kc.upper()}] {action.replace('-', ' ')}"
            t = font.render(label, True, (200, 200, 255))
            surf.blit(t, (pad, ky))

        screen.blit(surf, (panel_x, panel_y))


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


def _render_exposed_edges(screen, width, height, xmin, xmax, ymin, ymax, state, offset_x, offset_y, params=None):
    """Render only the strips of the screen not covered by the offset-blitted surface."""
    ox = int(offset_x)
    oy = int(offset_y)
    if ox == 0 and oy == 0:
        return

    if params is None:
        params = build_render_params(state, maxiter=state.get("max_iter", 256),
                                     use_julia=state.get("set_blend", 0.0) >= 1.0)

    scale_x = (xmax - xmin) / width
    scale_y = (ymax - ymin) / height

    if ox > 0:
        sx = xmin
        ex = xmin + ox * scale_x
        if sx < ex:
            surf, _ = render_to_surface(ox, height, sx, ex, ymin, ymax, state["max_iter"], state)
            screen.blit(surf, (0, 0))
    elif ox < 0:
        sx = xmax + ox * scale_x
        ex = xmax
        ow = -ox
        if sx < ex:
            surf, _ = render_to_surface(ow, height, sx, ex, ymin, ymax, state["max_iter"], state)
            screen.blit(surf, (width + ox, 0))

    if oy > 0:
        sy = ymax - oy * scale_y
        ey = ymax
        if sy < ey:
            surf, _ = render_to_surface(width, oy, xmin, xmax, sy, ey, state["max_iter"], state)
            screen.blit(surf, (0, 0))
    elif oy < 0:
        sy = ymin
        ey = ymin - oy * scale_y
        oh = -oy
        if sy < ey:
            surf, _ = render_to_surface(width, oh, xmin, xmax, sy, ey, state["max_iter"], state)
            screen.blit(surf, (0, height + oy))


def key_name_to_pygame(name):
    """Convert keybind name to (pygame key constant, modifier) tuple.
    Supports formats like 'ctrl+r', 'shift+r', 'r'."""
    name = name.strip()
    mod = None
    if "+" in name:
        parts = name.split("+")
        mod = parts[0].strip().lower()
        name = parts[1].strip()
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
    if mod:
        return (kc, _MOD_MAP.get(mod, 0))
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
    "stripe-up":   ("stripe_s",    1.0,  0,   101, True,  True,  None),
    "stripe-down": ("stripe_s",   -1.0,  0,   101, True,  True,  None),
    "step-up":     ("step_s",      1.0,  0.0, 100.0, False, True,  None),
    "step-down":   ("step_s",     -1.0,  0.0, 100.0, False, True,  None),
    "phase-up":    ("phase",       0.05, 0.0, 1.0,  True,  False, None),
    "phase-down":  ("phase",      -0.05, 0.0, 1.0,  True,  False, None),
    "light-angle-up":   ("light_angle",  0.1,  0.0, 1.0,  True,  False, None),
    "light-angle-down": ("light_angle", -0.1,  0.0, 1.0,  True,  False, None),
    "light-azim-up":    ("light_azim",   0.1,  0.0, 1.0,  True,  False, None),
    "light-azim-down":  ("light_azim",  -0.1,  0.0, 1.0,  True,  False, None),
    "light-i-up":       ("light_i",      0.05, 0.0, 1.0,  True,  False, None),
    "light-i-down":     ("light_i",     -0.05, 0.0, 1.0,  True,  False, None),
    "k-amb-up":    ("k_ambiant",   0.05, 0.0, 1.0,  True,  False, None),
    "k-amb-down":  ("k_ambiant",  -0.05, 0.0, 1.0,  True,  False, None),
    "k-diff-up":   ("k_diffuse",   0.05, 0.0, 1.0,  True,  False, None),
    "k-diff-down": ("k_diffuse",  -0.05, 0.0, 1.0,  True,  False, None),
    "k-spec-up":   ("k_specular",  0.05, 0.0, 1.0,  True,  False, None),
    "k-spec-down": ("k_specular", -0.05, 0.0, 1.0,  True,  False, None),
    "shininess-up":   ("shininess",  2.0,  0.0, 100.0, False, False, None),
    "shininess-down": ("shininess", -2.0,  0.0, 100.0, False, False, None),
    "hue0-up":   (None,  0.02, 0.0, 1.0,  True,  False, 0),
    "hue0-down": (None, -0.02, 0.0, 1.0,  True,  False, 0),
    "hue1-up":   (None,  0.02, 0.0, 1.0,  True,  False, 1),
    "hue1-down": (None, -0.02, 0.0, 1.0,  True,  False, 1),
    "sat-up":    (None,  0.02, 0.0, 1.0,  True,  False, 2),
    "sat-down":  (None, -0.02, 0.0, 1.0,  True,  False, 2),
    "blend-up":  ("set_blend",  0.05, 0.0, 1.0,  False, False, None),
    "blend-down": ("set_blend", -0.05, 0.0, 1.0,  False, False, None),
}


def _apply_step(state, action, mult):
    """Apply a numeric step action to state using the current shift multiplier."""
    key, delta, lo, hi, cyclic, to_int, rgb_idx = _NUMERIC_STEPS[action]
    if rgb_idx is not None:
        cur = state["rgb_thetas"][rgb_idx]
    else:
        cur = state[key]
    if cyclic:
        new_val = step_val_cyclic(cur, delta, lo, hi, mult)
    else:
        new_val = step_val(cur, delta, lo, hi, mult)
    if to_int:
        new_val = int(new_val)
    if rgb_idx is not None:
        state["rgb_thetas"][rgb_idx] = new_val
    else:
        state[key] = new_val


def _reset_to_defaults(state):
    """Reset all visual and interaction settings to their defaults."""
    state["rgb_thetas"] = list(COLOR_THETAS[DEFAULT_PALETTE_INDEX])
    state["phase"] = DEFAULT_PHASE
    state["stripe_s"] = DEFAULT_STRIPE_S
    state["stripe_sig"] = DEFAULT_STRIPE_SIG
    state["step_s"] = DEFAULT_STEP_S
    state["light_angle"] = DEFAULT_LIGHT_ANGLE
    state["light_azim"] = DEFAULT_LIGHT_AZIM
    state["light_i"] = DEFAULT_LIGHT_I
    state["k_ambiant"] = DEFAULT_K_AMBIANT
    state["k_diffuse"] = DEFAULT_K_DIFFUSE
    state["k_specular"] = DEFAULT_K_SPECULAR
    state["shininess"] = DEFAULT_SHININESS
    state["smooth"] = True
    state["use_gpu"] = False
    state["orbit_point"] = list(DEFAULT_ORBIT_POINT)
    state["show_orbits"] = True
    state["show_orbits_m"] = True
    state["show_orbits_j"] = True
    state["show_orbit_lines_m"] = DEFAULT_ORBIT_LINE_M
    state["show_orbit_lines_j"] = DEFAULT_ORBIT_LINE_J
    state["show_c_point"] = DEFAULT_SHOW_C_POINT
    state["c_point_size"] = DEFAULT_C_POINT_SIZE
    state["c_point_color"] = list(DEFAULT_C_POINT_COLOR)
    state["orbit_rgb_thetas"] = list(DEFAULT_ORBIT_RGB_THETAS)
    state["orbit_max_iter"] = 200
    state["orbit_point_size"] = 3
    state["fxaa"] = False
    state["palette_index"] = DEFAULT_PALETTE_INDEX
    state["set_blend"] = DEFAULT_SET_BLEND
    state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
    state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
    state["orbit_point"] = list(DEFAULT_ORBIT_POINT_M)  # back-compat
    state["julia_c"] = list(DEFAULT_JULIA_C)


def run_render_mode(settings, cli_iter=None, cli_color=None, cli_gpu=False, cli_no_gpu=False, cli_julia=False, cli_blend=None):
    max_iter = cli_iter if cli_iter is not None else get_persistent_setting(settings, "iteration-max", cast=int, default=128)
    if cli_no_gpu:
        use_gpu = False
    else:
        use_gpu = cli_gpu or (settings.get("use-gpu", "false").lower() == "true")
    init_width = get_persistent_setting(settings, "width", cast=int, default=800)
    init_height = get_persistent_setting(settings, "height", cast=int, default=600)

    keybinds = {k: get_keybind(settings, k, v) for k, v in DEFAULT_KEYBINDS.items()}

    key_map = {}
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.5, 1.5
    xmin, xmax, ymin, ymax = fix_aspect_ratio(xmin, xmax, ymin, ymax, init_width, init_height)

    pygame.init()
    screen = pygame.display.set_mode((init_width, init_height), pygame.RESIZABLE)
    pygame.display.set_caption("Mandelbrot/Julia — drag to pan, scroll to zoom, M for info")

    for action, name in keybinds.items():
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

    persistent = get_persistent_setting(settings, "persistent",
                                        cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                        default=True)

    state = {
        "max_iter": max_iter,
        "rgb_thetas": [
            get_persistent_setting(settings, "hue_0", cast=float, default=DEFAULT_RGB_THETAS[0]),
            get_persistent_setting(settings, "hue_1", cast=float, default=DEFAULT_RGB_THETAS[1]),
            get_persistent_setting(settings, "sat", cast=float, default=DEFAULT_RGB_THETAS[2]),
        ],
        "phase": get_persistent_setting(settings, "phase", cast=float, default=DEFAULT_PHASE),
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
                                                 cast=int, default=DEFAULT_PALETTE_INDEX),
        "orbit_point": [
            get_persistent_setting(settings, "orbit_x", cast=float, default=DEFAULT_ORBIT_POINT[0]),
            get_persistent_setting(settings, "orbit_y", cast=float, default=DEFAULT_ORBIT_POINT[1]),
        ],
        "orbit_point_m": [
            get_persistent_setting(settings, "orbit-mx", cast=float, default=DEFAULT_ORBIT_POINT_M[0]),
            get_persistent_setting(settings, "orbit-my", cast=float, default=DEFAULT_ORBIT_POINT_M[1]),
        ],
        "orbit_point_j": [
            get_persistent_setting(settings, "orbit-jx", cast=float, default=DEFAULT_ORBIT_POINT_J[0]),
            get_persistent_setting(settings, "orbit-jy", cast=float, default=DEFAULT_ORBIT_POINT_J[1]),
        ],
        "show_orbits": get_persistent_setting(settings, "show-orbits",
                                               cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                               default=True),
        "show_orbits_m": get_persistent_setting(settings, "show-orbits-m",
                                                cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                default=True),
        "show_orbits_j": get_persistent_setting(settings, "show-orbits-j",
                                                cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                default=True),
        "orbit_max_iter": get_persistent_setting(settings, "orbit-max-iter", cast=int, default=200),
        "orbit_point_size": get_persistent_setting(settings, "orbit-point-size", cast=int, default=3),
        "set_blend": get_persistent_setting(settings, "set-blend",
                                           cast=float, default=DEFAULT_SET_BLEND),
        "julia_c": [
            get_persistent_setting(settings, "julia-cx", cast=float, default=DEFAULT_JULIA_C[0]),
            get_persistent_setting(settings, "julia-cy", cast=float, default=DEFAULT_JULIA_C[1]),
        ],
        "show_c_point": get_persistent_setting(settings, "show-c-point",
                                              cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                              default=DEFAULT_SHOW_C_POINT),
        "c_point_size": get_persistent_setting(settings, "c-point-size", cast=int, default=DEFAULT_C_POINT_SIZE),
        "c_point_color": [
            get_persistent_setting(settings, "c-point-color-r", cast=float, default=DEFAULT_C_POINT_COLOR[0]),
            get_persistent_setting(settings, "c-point-color-g", cast=float, default=DEFAULT_C_POINT_COLOR[1]),
            get_persistent_setting(settings, "c-point-color-b", cast=float, default=DEFAULT_C_POINT_COLOR[2]),
        ],
        "orbit_rgb_thetas": [
            get_persistent_setting(settings, "orbit-hue-0", cast=float, default=DEFAULT_ORBIT_RGB_THETAS[0]),
            get_persistent_setting(settings, "orbit-hue-1", cast=float, default=DEFAULT_ORBIT_RGB_THETAS[1]),
            get_persistent_setting(settings, "orbit-sat", cast=float, default=DEFAULT_ORBIT_RGB_THETAS[2]),
        ],
        "show_orbit_lines_m": get_persistent_setting(settings, "show-orbit-lines-m",
                                                      cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                      default=DEFAULT_ORBIT_LINE_M),
        "show_orbit_lines_j": get_persistent_setting(settings, "show-orbit-lines-j",
                                                      cast=lambda s: str(s).strip().lower() in ("true", "1", "yes"),
                                                      default=DEFAULT_ORBIT_LINE_J),
    }
    state["rgb_thetas"] = list(COLOR_THETAS[state["palette_index"]])

    if cli_color is not None:
        _color_map = {name: i for i, name in enumerate(PALETTE_NAMES)}
        state["palette_index"] = _color_map[cli_color]
        state["rgb_thetas"] = list(COLOR_THETAS[state["palette_index"]])
    if cli_blend is not None:
        state["set_blend"] = max(0.0, min(1.0, round(float(cli_blend), 4)))
    elif cli_julia:
        state["set_blend"] = 1.0

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
    force_full_render = False
    last_mouse_pos = (0, 0)
    needs_render = True
    surface = None
    interacting = False
    interact_timer = 0
    INTERACT_SETTLE = 100
    drag_offset_x = 0.0
    drag_offset_y = 0.0
    last_render_xmin = None
    last_render_xmax = None
    last_render_ymin = None
    last_render_ymax = None
    _START_TIME = time.time()

    if _PERF_MONITOR:
        _PERF_MONITOR.start()

    overlay = MenuOverlay()
    overlay_font = _font if _font else pygame.font.SysFont("monospace", 18)

    clock = pygame.time.Clock()
    running = True
    frame_num = 0
    _last_caption = None

    while running:
        _t_frame_start = time.perf_counter()
        _t_render = 0.0
        _used_gpu = False
        still_interacting = False
        events = pygame.event.get()
        _dbg(f"poll_events count={len(events)}")
        for event in events:
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                pass

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if clicking on menu overlay first
                    if overlay.active and overlay.handle_click(event.pos, state):
                        _RENDER_CACHE.clear()
                        force_full_render = True
                        needs_render = True
                        continue
                    _dbg(f"MOUSEBUTTONDOWN {event.pos}")
                    mpx, mpy = event.pos
                    # Check if clicking on any of the 3 draggable points
                    # Priority: orbit point that matches current blend (more visible), then c_J, then others
                    click_targets = []
                    set_blend = state.get("set_blend", 0.0)
                    mb_alpha = 1.0 - set_blend

                    # Mandelbrot orbit point (blue)
                    mx, my = state.get("orbit_point_m", state.get("orbit_point", DEFAULT_ORBIT_POINT_M))
                    px = int((mx - xmin) / (xmax - xmin) * width) if (xmax - xmin) > 0 else width // 2
                    py = int((ymax - my) / (ymax - ymin) * height) if (ymax - ymin) > 0 else height // 2
                    dist_mb = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    click_targets.append((dist_mb, "mandelbrot", mb_alpha))

                    # Julia orbit point (red)
                    jx, jy = state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)
                    px = int((jx - xmin) / (xmax - xmin) * width) if (xmax - xmin) > 0 else width // 2
                    py = int((ymax - jy) / (ymax - ymin) * height) if (ymax - ymin) > 0 else height // 2
                    dist_ju = math.sqrt((mpx - px) ** 2 + (mpy - py) ** 2)
                    click_targets.append((dist_ju, "julia", set_blend))

                    # Julia constant c_J (green)
                    cx, cy = state.get("julia_c", DEFAULT_JULIA_C)
                    px = int((cx - xmin) / (xmax - xmin) * width) if (xmax - xmin) > 0 else width // 2
                    py = int((ymax - cy) / (ymax - ymin) * height) if (ymax - ymin) > 0 else height // 2
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
                    _dbg(f"MOUSEBUTTONUP {event.pos} dragging=False")
                    orbit_drag = False
                    orbit_drag_mode = None
                    state["orbit_drag_mode"] = None
                    state["orbit_hover"] = False
                    dragging = False
                    drag_offset_x = 0.0
                    drag_offset_y = 0.0
                    needs_render = True

            elif event.type == pygame.MOUSEMOTION:
                if orbit_drag:
                    mx, my = event.pos
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
                        state["orbit_point"] = [new_sx, new_sy]  # back-compat
                    _dbg(f"ORBIT_DRAG ({orbit_drag_mode}) to ({new_sx:.4f}, {new_sy:.4f})")
                    needs_render = True
                elif dragging:
                    _dbg(f"MOUSEMOTION {event.pos} drag")
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    last_mouse_pos = event.pos

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
                zoom_factor = 0.8 if event.y > 0 else 1.25

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
                bindings = key_map.get(event.key)
                if not bindings:
                    continue
                mods = pygame.key.get_mods()
                result = None
                for action, mod_val in bindings:
                    if mod_val != 0 and bool(mods & mod_val):
                        result = (action, mod_val)
                        break
                    if mod_val == 0 and result is None:
                        result = (action, mod_val)
                if result is None:
                    continue
                action, mod_val = result

                _mult = 10.0 if is_shift_held() else 1.0

                if action in _NUMERIC_STEPS:
                    _apply_step(state, action, _mult)
                    _RENDER_CACHE.clear()
                    force_full_render = True
                    needs_render = True

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
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-info":
                    overlay.toggle()
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
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-fxaa":
                    state["fxaa"] = not state.get("fxaa", False)
                    _RENDER_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "toggle-orbits":
                    new_val = not state.get("show_orbits", True)
                    state["show_orbits"] = new_val
                    state["show_orbits_m"] = new_val
                    state["show_orbits_j"] = new_val
                    needs_render = True

                elif action == "toggle-orbits-m":
                    state["show_orbits_m"] = not state.get("show_orbits_m", True)
                    needs_render = True

                elif action == "toggle-orbits-j":
                    state["show_orbits_j"] = not state.get("show_orbits_j", True)
                    state["show_orbits"] = state.get("show_orbits_m", True) or state.get("show_orbits_j", True)
                    needs_render = True

                elif action == "toggle-orbit-lines-m":
                    state["show_orbit_lines_m"] = not state.get("show_orbit_lines_m", True)
                    needs_render = True

                elif action == "toggle-orbit-lines-j":
                    state["show_orbit_lines_j"] = not state.get("show_orbit_lines_j", True)
                    needs_render = True

                elif action == "toggle-c-point":
                    state["show_c_point"] = not state.get("show_c_point", True)
                    needs_render = True

                elif action == "reset-orbit-point":
                    state["orbit_point_m"] = list(DEFAULT_ORBIT_POINT_M)
                    state["orbit_point_j"] = list(DEFAULT_ORBIT_POINT_J)
                    state["orbit_point"] = list(DEFAULT_ORBIT_POINT_M)
                    state["julia_c"] = list(DEFAULT_JULIA_C)
                    needs_render = True

                elif action == "swap-orbit-point":
                    state["orbit_point_m"], state["orbit_point_j"] = \
                        list(state["orbit_point_j"]), list(state["orbit_point_m"])
                    state["orbit_point"] = list(state["orbit_point_m"])
                    needs_render = True

                elif action == "cycle-palette":
                    state["palette_index"] = (state.get("palette_index", 0) + 1) % len(COLOR_THETAS)
                    t = COLOR_THETAS[state["palette_index"]]
                    state["rgb_thetas"] = [t[0], t[1], t[2]]
                    _RENDER_CACHE.clear()
                    force_full_render = True
                    needs_render = True

                elif action == "quit":
                    running = False

        _sw, _sh = screen.get_size()
        if (_sw, _sh) != (width, height):
            _dbg(f"WINDOW_RESIZE {width}x{height} -> {_sw}x{_sh}")
            width, height = _sw, _sh
            xmin, xmax, ymin, ymax = fix_aspect_ratio(
                xmin, xmax, ymin, ymax, width, height)
            interacting = True
            interact_timer = pygame.time.get_ticks()
            needs_render = True
            drag_offset_x = 0.0
            drag_offset_y = 0.0

        if needs_render:
            still_interacting = interacting and (pygame.time.get_ticks() - interact_timer < INTERACT_SETTLE)
            do_offset_blit = (still_interacting or dragging) and not force_full_render

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
                # No surface yet or negligible offset: full render
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
                # Idle render (not interacting)
                interacting = False
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

        backend = "CUDA" if (state["use_gpu"] and _CUDA_AVAILABLE) else "CPU"
        _t_blit0 = time.perf_counter()
        if needs_render and dragging and (abs(drag_offset_x) > 0.1 or abs(drag_offset_y) > 0.1):
            screen.blit(surface, (int(drag_offset_x), int(drag_offset_y)))
        else:
            screen.blit(surface, (0, 0))
        _t_blit = time.perf_counter() - _t_blit0
        # Draw orbit points on top of the rendered surface
        _draw_orbit(screen, state, xmin, xmax, ymin, ymax, width, height)
        if overlay.active:
            overlay.draw(screen, overlay_font, state, keybinds)
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
            perf_str = (f"[perf] frame={frame_num} total={(time.perf_counter()-_t_frame_start)*1000:.1f}ms "
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
        frame_num += 1
        clock.tick(60)

    settings["width"] = str(width)
    settings["height"] = str(height)
    _save_map = [
        ("iteration-max", state["max_iter"]),
        ("use-gpu", state["use_gpu"]),
        ("phase", state["phase"]),
        ("hue_0", state["rgb_thetas"][0]),
        ("hue_1", state["rgb_thetas"][1]),
        ("sat", state["rgb_thetas"][2]),
        ("light_angle", state["light_angle"]),
        ("light_azim", state["light_azim"]),
        ("light_i", state["light_i"]),
        ("k_ambiant", state["k_ambiant"]),
        ("k_diffuse", state["k_diffuse"]),
        ("k_specular", state["k_specular"]),
        ("shininess", state["shininess"]),
        ("stripe_s", state["stripe_s"]),
        ("step_s", state["step_s"]),
        ("smooth", state["smooth"]),
        ("fxaa", state.get("fxaa", False)),
        ("show-orbits", state.get("show_orbits", True)),
        ("show-orbits-m", state.get("show_orbits_m", True)),
        ("show-orbits-j", state.get("show_orbits_j", True)),
        ("set-blend", state.get("set_blend", 0.0)),
        ("julia-cx", state.get("julia_c", DEFAULT_JULIA_C)[0]),
        ("julia-cy", state.get("julia_c", DEFAULT_JULIA_C)[1]),
        ("orbit-mx", state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[0]),
        ("orbit-my", state.get("orbit_point_m", DEFAULT_ORBIT_POINT_M)[1]),
        ("orbit-jx", state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[0]),
        ("orbit-jy", state.get("orbit_point_j", DEFAULT_ORBIT_POINT_J)[1]),
        ("orbit_x", state.get("orbit_point", DEFAULT_ORBIT_POINT_M)[0]),
        ("orbit_y", state.get("orbit_point", DEFAULT_ORBIT_POINT_M)[1]),
        ("orbit-max-iter", state["orbit_max_iter"]),
        ("orbit-point-size", state.get("orbit_point_size", 3)),
        ("c-point-size", state.get("c_point_size", 5)),
        ("show-c-point", state.get("show_c_point", True)),
        ("c-point-color-r", state.get("c_point_color", DEFAULT_C_POINT_COLOR)[0]),
        ("c-point-color-g", state.get("c_point_color", DEFAULT_C_POINT_COLOR)[1]),
        ("c-point-color-b", state.get("c_point_color", DEFAULT_C_POINT_COLOR)[2]),
        ("orbit-hue-0", state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[0]),
        ("orbit-hue-1", state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[1]),
        ("orbit-sat", state.get("orbit_rgb_thetas", DEFAULT_ORBIT_RGB_THETAS)[2]),
        ("show-orbit-lines-m", state.get("show_orbit_lines_m", True)),
        ("show-orbit-lines-j", state.get("show_orbit_lines_j", True)),
        ("palette-index", state.get("palette_index", 0)),
    ]
    for key, val in _save_map:
        settings[key] = str(val)
    for action, kc in keybinds.items():
        settings[f"keybind.{action}"] = kc
    print(f"[debug] use_gpu={state['use_gpu']} cuda_available={_CUDA_AVAILABLE} "
          f"final_size={width}x{height} iter={state['max_iter']} frames={frame_num}",
          file=sys.stderr)
    if persistent:
        save_settings(SETTINGS_FILE, settings)

    if _PERF_MONITOR:
        _PERF_MONITOR.stop()

    if _dbg_file is not None:
        _dbg("exit")
        _dbg_file.close()
    pygame.quit()


# --- CLI args ---
# --- Mode selector ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=None, help="Max iterations")
    parser.add_argument("--color", type=str, default=None,
                        choices=["fire", "deep-sea", "arctic", "ocean", "twilight",
                                 "magenta", "aurora", "forest", "grayscale",
                                 "monochrome", "sunset", "amber", "ice",
                                 "jade", "copper", "violet"], help="Color palette preset")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU rendering")
    parser.add_argument("--julia", action="store_true", help="Render Julia set (equivalent to --blend 1.0)")
    parser.add_argument("--blend", type=float, default=None, help="Blend factor: 0=MB only, 1=Julia only")
    args = parser.parse_args()
    settings = load_settings(SETTINGS_FILE)
    # --julia is backward compatible: equivalent to --blend 1.0
    if args.julia and args.blend is None:
        args.blend = 1.0
    run_render_mode(settings, cli_iter=args.iter, cli_color=args.color,
                    cli_gpu=args.gpu, cli_no_gpu=args.no_gpu, cli_julia=args.julia,
                    cli_blend=args.blend)
