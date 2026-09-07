import math
import os
import numpy as np
from numba import njit, prange
import pygame
import sys
import argparse
import time


try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    _CUDA_AVAILABLE = False

from numba import config
print("Numba threads:", config.NUMBA_NUM_THREADS, "| CUDA:", _CUDA_AVAILABLE)

_dbg_file = None


def _dbg(msg):
    if _dbg_file is not None:
        _dbg_file.write(f"{time.time():.3f} {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        _dbg_file.flush()


DEFAULT_NCYCLE = 32
DEFAULT_STRIPE_S = 0.0
DEFAULT_STRIPE_SIG = 0.9
DEFAULT_STEP_S = 0.0
DEFAULT_LIGHT_ANGLE = 0.46  # 0.125/(2*pi) normalized
DEFAULT_LIGHT_AZIM = 0.5   # 0.5/(pi/2) normalized
DEFAULT_LIGHT_I = 0.75
DEFAULT_K_AMBIANT = 0.2
DEFAULT_K_DIFFUSE = 0.5
DEFAULT_K_SPECULAR = 0.5
DEFAULT_SHININESS = 20.0
DEFAULT_RGB_THETAS = [0.85, 0.0, 0.15]  # fire
DEFAULT_PHASE = 0.0
NCOL = 2 ** 12

COLOR_THETAS = [
    [0.85, 0.0, 0.15],   # fire
    [0.0, 0.15, 0.25],   # cool
    [0.1, 0.5, 0.9],     # deep
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
    val = settings.get(f"keybind.{name}", default_key).lower()
    return val


def get_persistent_setting(settings, key, cast=str, default=None, prompt=None):
    if key in settings:
        try:
            return cast(settings[key])
        except ValueError:
            print(f"Warning: '{key}' = '{settings[key]}' is invalid, will re-prompt.")
    prompt_text = prompt or f"Enter value for '{key}'"
    if default is not None:
        prompt_text += f" [default: {default}]"
    prompt_text += ": "
    raw = input(prompt_text).strip()
    value = default if raw == "" and default is not None else cast(raw)
    settings[key] = str(value)
    return value


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
        zx, zy = zx*zx - zy*zy + cx, 2*zx*zy + cy
        orbit_x[i] = zx
        orbit_y[i] = zy
        if zx*zx + zy*zy > 4:
            return orbit_x[:i+1], orbit_y[:i+1], True
    return orbit_x, orbit_y, False


def make_colortable(rgb_thetas):
    x = np.linspace(0.0, 1.0, NCOL)
    y = np.column_stack(((x + rgb_thetas[0]) * 2 * math.pi,
                         (x + rgb_thetas[1]) * 2 * math.pi,
                         (x + rgb_thetas[2]) * 2 * math.pi))
    val = (0.5 + 0.5 * np.sin(y)).astype(np.float32)
    return val


@njit
def overlay(x, y, gamma):
    if (2 * y) < 1:
        out = 2 * x * y
    else:
        out = 1 - 2 * (1 - x) * (1 - y)
    return out * gamma + x * (1 - gamma)


@njit
def blinn_phong(normal, light):
    normal = normal / abs(normal)
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
def smooth_iter(c, maxiter, stripe_s, stripe_sig):
    esc_radius_2 = 10.0 ** 10
    z = 0j
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a = 0.0
    dz = 1 + 0j
    for n in range(maxiter):
        dz = dz * 2 * z + 1
        z = z * z + c
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


@njit
def color_pixel(niter, stripe_a, step_s, dem, normal, colortable, ncycle, light):
    ncol = colortable.shape[0] - 1
    niter = math.sqrt(niter) % ncycle / ncycle
    col_i = round(niter * ncol)

    bright = blinn_phong(normal, light)
    dem = -math.log(dem) / 12
    dem = 1.0 / (1.0 + math.exp(-10 * ((2 * dem - 1) / 2)))

    nshader = 0
    shader = 0.0
    if stripe_a > 0:
        nshader += 1
        shader += stripe_a
    if step_s > 0:
        # step_s controls color quantization: higher = more bands
        n_steps = max(1.0, step_s)
        quantized = math.floor(niter * n_steps) / n_steps
        x = (niter - quantized) * n_steps
        col_i = round(quantized * ncol)
        light_step = 6 * (1 - x ** 5 - (1 - x) ** 100) / 10
        x2 = (niter - quantized) * n_steps * 8
        light_step2 = 6 * (1 - (x2 - math.floor(x2)) ** 5 - (1 - (x2 - math.floor(x2))) ** 30) / 10
        light_step = overlay(light_step2, light_step, 1)
        nshader += 1
        shader += light_step
    if nshader > 0:
        bright = overlay(bright, shader / nshader, 1) * (1 - dem) + dem * bright

    r = overlay(colortable[col_i, 0], bright, 1)
    g = overlay(colortable[col_i, 1], bright, 1)
    b = overlay(colortable[col_i, 2], bright, 1)
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0
    if g < 0.0:
        g = 0.0
    elif g > 1.0:
        g = 1.0
    if b < 0.0:
        b = 0.0
    elif b > 1.0:
        b = 1.0
    return (r, g, b)


@cuda.jit(device=True)
def _overlay_cuda(x, y, gamma):
    if (2 * y) < 1:
        out = 2 * x * y
    else:
        out = 1 - 2 * (1 - x) * (1 - y)
    return out * gamma + x * (1 - gamma)


@cuda.jit(device=True)
def _blinn_phong_cuda(normal_re, normal_im, light):
    mag = math.sqrt(normal_re * normal_re + normal_im * normal_im)
    if mag == 0:
        return 0.0
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
    return bright


@cuda.jit(device=True)
def _color_pixel_cuda(niter, stripe_a, step_s, dem, nr, ni, colortable, ncol, light, ncycle):
    niter = math.sqrt(niter) % ncycle / ncycle
    col_i = int(round(niter * ncol))

    bright = _blinn_phong_cuda(nr, ni, light)
    dem = -math.log(dem) / 12
    dem = 1.0 / (1.0 + math.exp(-10 * ((2 * dem - 1) / 2)))

    nshader = 0
    shader = 0.0
    if stripe_a > 0:
        nshader += 1
        shader += stripe_a
    if step_s > 0:
        n_steps = max(1.0, step_s)
        quantized = math.floor(niter * n_steps) / n_steps
        x = (niter - quantized) * n_steps
        col_i = int(round(quantized * ncol))
        light_step = 6 * (1 - math.pow(x, 5) - math.pow(1 - x, 30)) / 10
        x2 = (niter - quantized) * n_steps * 8
        x2f = x2 - math.floor(x2)
        light_step2 = 6 * (1 - math.pow(x2f, 5) - math.pow(1 - x2f, 30)) / 10
        if 2 * light_step2 < 1:
            ls = 2 * light_step * light_step2
        else:
            ls = 1 - 2 * (1 - light_step) * (1 - light_step2)
        nshader += 1
        shader += ls
    if nshader > 0:
        bright = _overlay_cuda(bright, shader / nshader, 1) * (1 - dem) + dem * bright

    r = _overlay_cuda(colortable[col_i, 0], bright, 1)
    g = _overlay_cuda(colortable[col_i, 1], bright, 1)
    b = _overlay_cuda(colortable[col_i, 2], bright, 1)
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return (r, g, b)


@njit(parallel=True)
def compute_set_cpu(creal, cim, maxiter, colortable, ncycle,
                    stripe_s, stripe_sig, step_s, diag, light):
    xpixels = len(creal)
    ypixels = len(cim)
    mat = np.zeros((ypixels, xpixels, 3), dtype=np.float32)
    for x in prange(xpixels):
        for y in range(ypixels):
            niter, stripe_a, dem, normal = smooth_iter(
                complex(creal[x], cim[y]), maxiter, stripe_s, stripe_sig)
            if niter > 0:
                r, g, b = color_pixel(niter, stripe_a, step_s, dem / diag,
                                      normal, colortable, ncycle, light)
                mat[y, x, 0] = r
                mat[y, x, 1] = g
                mat[y, x, 2] = b
    return mat


if cuda is not None:
    @cuda.jit
    def compute_set_gpu(mat, xmin, xmax, ymin, ymax, maxiter, colortable,
                        ncycle, stripe_s, stripe_sig, step_s, diag, light):
        index = cuda.grid(1)
        x = index % mat.shape[1]
        y = index // mat.shape[1]
        if y < mat.shape[0] and x < mat.shape[1]:
            creal = xmin + x / (mat.shape[1] - 1) * (xmax - xmin)
            cim = ymin + y / (mat.shape[0] - 1) * (ymax - ymin)
            ncol = colortable.shape[0] - 1
            # Inline smooth_iter
            esc_radius_2 = 1e10
            zr, zi = 0.0, 0.0
            dzr, dzi = 1.0, 0.0
            niter = 0.0
            stripe_a = 0.0
            dem = 0.0
            normal_re, normal_im = 0.0, 0.0
            for n in range(maxiter):
                new_dzr = dzr * 2 * zr - dzi * 2 * zi + 1
                new_dzi = dzi * 2 * zr + dzr * 2 * zi
                dzr, dzi = new_dzr, new_dzi
                new_zr = zr * zr - zi * zi + creal
                new_zi = 2 * zr * zi + cim
                zr, zi = new_zr, new_zi
                if zr * zr + zi * zi > esc_radius_2:
                    modz = math.sqrt(zr * zr + zi * zi)
                    log_ratio = 2 * math.log(modz) / math.log(esc_radius_2)
                    smooth_i = 1 - math.log(log_ratio) / math.log(2)
                    niter = float(n + smooth_i)
                    dem = modz * math.log(modz) / math.sqrt(dzr * dzr + dzi * dzi) / 2
                    normal_re = zr * dzi - zi * dzr
                    normal_im = zr * dzr + zi * dzi
                    ndem = math.sqrt(dzr * dzr + dzi * dzi)
                    if ndem > 0:
                        normal_re = normal_re / ndem
                        normal_im = normal_im / ndem
                    break
            if niter > 0:
                cniter = math.sqrt(niter) % ncycle / ncycle
                col_i = int(round(cniter * ncol))
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
                dem = -math.log(dem) / 12
                dem = 1.0 / (1.0 + math.exp(-10 * ((2 * dem - 1) / 2)))
                nshader = 0
                shader = 0.0
                if stripe_a > 0:
                    nshader += 1
                    shader += stripe_a
                if step_s > 0:
                    n_steps = max(1.0, step_s)
                    quantized = math.floor(cniter * n_steps) / n_steps
                    x2 = (cniter - quantized) * n_steps
                    col_i = int(quantized * ncol)
                    light_step = 6 * (1 - math.pow(x2, 5) - math.pow(1 - x2, 30)) / 10
                    x8 = (cniter - quantized) * n_steps * 8
                    light_step2 = 6 * (1 - math.pow(x8 - math.floor(x8), 5) - math.pow(1 - (x8 - math.floor(x8)), 30)) / 10
                    if 2 * light_step2 < 1:
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


def build_render_params(state):
    rgb_thetas = list(state["rgb_thetas"])
    phase = state["phase"]
    colortable = make_colortable(np.array([t + phase for t in rgb_thetas], dtype=np.float64))
    light = np.array([
        state["light_angle"] * 2 * math.pi,
        state["light_azim"] * math.pi / 2,
        state["light_i"],
        state["k_ambiant"],
        state["k_diffuse"],
        state["k_specular"],
        state["shininess"],
    ], dtype=np.float64)
    return {
        "colortable": colortable,
        "ncy": math.sqrt(DEFAULT_NCYCLE),
        "stripe_s": state["stripe_s"],
        "stripe_sig": state["stripe_sig"],
        "step_s": state["step_s"],
        "light": light,
        "use_gpu": state["use_gpu"] and _CUDA_AVAILABLE,
    }


def compute_image(width, height, xmin, xmax, ymin, ymax, maxiter, params):
    colortable = params["colortable"]
    ncycle = params["ncy"]
    stripe_s = params["stripe_s"]
    stripe_sig = params["stripe_sig"]
    step_s = params["step_s"]
    light = params["light"]
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    if params["use_gpu"] and cuda is not None:
        try:
            mat = cuda.device_array((height, width, 3), dtype=np.float32)
            if "_colortable_d" not in params:
                params["_colortable_d"] = cuda.to_device(colortable)
            if "_light_d" not in params:
                params["_light_d"] = cuda.to_device(light)
            colortable_d = params["_colortable_d"]
            light_d = params["_light_d"]
            npixels = width * height
            nthread = 256
            nblock = math.ceil(npixels / nthread)
            compute_set_gpu[nblock, nthread](
                mat, xmin, xmax, ymin, ymax, maxiter, colortable_d, ncycle,
                stripe_s, stripe_sig, step_s, diag, light_d)
            cuda.synchronize()
            return mat.copy_to_host()
        except Exception as e:
            print(f"[gpu] falling back to CPU: {e}", file=sys.stderr)

    creal = np.linspace(xmin, xmax, width)
    cim = np.linspace(ymin, ymax, height)
    return compute_set_cpu(creal, cim, maxiter, colortable, ncycle,
                           stripe_s, stripe_sig, step_s, diag, light)


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
_PERF_STATS = {"render_count": 0, "total_render_ms": 0.0}


def render_to_surface(width, height, xmin, xmax, ymin, ymax, max_iter, state):
    key = (tuple(state["rgb_thetas"]), state["phase"], state["use_gpu"] and _CUDA_AVAILABLE)
    params = _RENDER_CACHE.get(key)
    if params is None:
        params = build_render_params(state)
        _RENDER_CACHE[key] = params
    _t0 = time.perf_counter()
    mat = compute_image(width, height, xmin, xmax, ymin, ymax, max_iter, params)
    _t_compute = (time.perf_counter() - _t0) * 1000
    mat = mat[::-1, :, :]
    rgb = (mat * 255.0).astype(np.uint8)
    rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    surf = pygame.surfarray.make_surface(rgb_t)
    _t_total = (time.perf_counter() - _t0) * 1000
    _PERF_STATS["render_count"] += 1
    _PERF_STATS["total_render_ms"] += _t_total
    if _PERF_STATS["render_count"] % 10 == 0:
        print(f"[render] n={_PERF_STATS['render_count']} "
              f"compute={_t_compute:.0f}ms post={_t_total-_t_compute:.0f}ms "
              f"avg={_PERF_STATS['total_render_ms']/_PERF_STATS['render_count']:.0f}ms "
              f"gpu={state['use_gpu'] and _CUDA_AVAILABLE}", file=sys.stderr)
    return surf, state["use_gpu"] and _CUDA_AVAILABLE


class InfoOverlay:
    """Simple on-screen info/keybinds overlay toggled with 'm' key."""
    def __init__(self):
        self.active = False

    def toggle(self):
        self.active = not self.active

    def draw(self, screen, font, state, keybinds):
        if not self.active:
            return
        lines = [
            "CONTROLS (configurable in .yaml) — hold SHIFT for 10x steps",
            f"  [{keybinds['toggle-info'].upper()}] toggle info",
            f"  [{keybinds['reset-view'].upper()}] reset view",
            f"  [{keybinds['quit'].upper()}] quit",
            f"  [{keybinds['toggle-gpu'].upper()}] toggle GPU/CPU",
            f"  [{keybinds['cycle-color'].upper()}] cycle color",
            f"  [{keybinds['zoom-in'].upper()}]/[{keybinds['zoom-out'].upper()}] zoom",
            f"  [{keybinds['iter-up'].upper()}]/[{keybinds['iter-down'].upper()}] iter up/down",
            f"  [{keybinds['stripe-up'].upper()}]/[{keybinds['stripe-down'].upper()}] stripe_s",
            f"  [{keybinds['step-up'].upper()}]/[{keybinds['step-down'].upper()}] step_s",
            f"  [{keybinds['light-i-up'].upper()}]/[{keybinds['light-i-down'].upper()}] light_i    (1,0)",
            f"  [{keybinds['light-angle-up'].upper()}]/[{keybinds['light-angle-down'].upper()}] light_angle",
            f"  [{keybinds['light-azim-up'].upper()}]/[{keybinds['light-azim-down'].upper()}] light_azim",
            f"  [{keybinds['k-amb-up'].upper()}]/[{keybinds['k-amb-down'].upper()}]  k_amb     (0.5)",
            f"  [{keybinds['k-diff-up'].upper()}]/[{keybinds['k-diff-down'].upper()}]  k_diff    (0.5)",
            f"  [{keybinds['k-spec-up'].upper()}]/[{keybinds['k-spec-down'].upper()}]  k_spec    (0.5)",
            f"  [{keybinds['shininess-up'].upper()}]/[{keybinds['shininess-down'].upper()}] shininess  (2)",
            f"  [{keybinds['phase-up'].upper()}]/[{keybinds['phase-down'].upper()}]   phase      (0.05)",
            f"  [{keybinds['rgb-r-up'].upper()}]/[{keybinds['rgb-r-down'].upper()}]    rgb_r      (0.02)",
            f"  [{keybinds['rgb-g-up'].upper()}]/[{keybinds['rgb-g-down'].upper()}]    rgb_g      (0.02)",
            f"  [{keybinds['rgb-b-up'].upper()}]/[{keybinds['rgb-b-down'].upper()}]    rgb_b      (0.02)",
            "---",
            f"  iter  = {state['max_iter']}",
            f"  stripe_s={state['stripe_s']:.1f}",
            f"  step_s  = {state['step_s']:.1f}",
            f"  phase   = {state['phase']:.2f} (0~1)",
            f"  rgb     = {state['rgb_thetas']}",
            f"  light_angle = {state['light_angle']:.2f}",
            f"  light_azim  = {state['light_azim']:.2f}",
            f"  light_i     = {state['light_i']:.2f}",
            f"  k_amb       = {state['k_ambiant']:.2f}",
            f"  k_diff      = {state['k_diffuse']:.2f}",
            f"  k_spec      = {state['k_specular']:.2f}",
            f"  shininess   = {state['shininess']:.0f}",
            f"  palette = {['fire','cool','deep'][next((i for i, p in enumerate(COLOR_THETAS) if [round(t,2) for t in state['rgb_thetas']] == [round(t,2) for t in p]), 0)]}",
            f"  {'GPU' if state['use_gpu'] and _CUDA_AVAILABLE else 'CPU'}",
        ]
        bx, by = 40, 40
        max_w = max(font.size(line)[0] for line in lines)
        surf = pygame.Surface((max_w + 20, len(lines) * 24 + 20), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))
        screen.blit(surf, (bx, by))
        pygame.draw.rect(screen, (200, 200, 200), (bx, by, max_w + 20, len(lines) * 24 + 20), 1)
        for i, line in enumerate(lines):
            color = (255, 255, 100) if i < 10 else (200, 200, 200)
            txt = font.render(line, True, color)
            screen.blit(txt, (bx + 10, by + 10 + i * 24))


def next_pow2(n):
    if n < 1:
        return 2
    return 2 ** n.bit_length()


def prev_pow2(n):
    if n <= 1:
        return 1
    return 2 ** (n.bit_length() - 2)


def _render_exposed_edges(screen, width, height, xmin, xmax, ymin, ymax, state, offset_x, offset_y):
    """Render only the strips of the screen not covered by the offset-blitted surface."""
    ox = int(round(offset_x))
    oy = int(round(offset_y))
    if ox == 0 and oy == 0:
        return
    scale_x = (xmax - xmin) / width
    scale_y = (ymax - ymin) / height

    if ox > 0:
        sx = xmin
        ex = xmin + ox * scale_x
        if sx < ex:
            mat = compute_image(ox, height, sx, ex, ymin, ymax, state["max_iter"],
                               build_render_params(state))
            mat = mat[::-1, :, :]
            rgb = (mat * 255.0).astype(np.uint8)
            rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
            strip = pygame.surfarray.make_surface(rgb_t)
            screen.blit(strip, (0, 0))
    elif ox < 0:
        sx = xmax + ox * scale_x
        ex = xmax
        ow = -ox
        if sx < ex:
            mat = compute_image(ow, height, sx, ex, ymin, ymax, state["max_iter"],
                               build_render_params(state))
            mat = mat[::-1, :, :]
            rgb = (mat * 255.0).astype(np.uint8)
            rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
            strip = pygame.surfarray.make_surface(rgb_t)
            screen.blit(strip, (width + ox, 0))

    if oy > 0:
        sy = ymax - oy * scale_y
        ey = ymax
        if sy < ey:
            mat = compute_image(width, oy, xmin, xmax, sy, ey, state["max_iter"],
                               build_render_params(state))
            mat = mat[::-1, :, :]
            rgb = (mat * 255.0).astype(np.uint8)
            rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
            strip = pygame.surfarray.make_surface(rgb_t)
            screen.blit(strip, (0, 0))
    elif oy < 0:
        sy = ymin
        ey = ymin - oy * scale_y
        oh = -oy
        if sy < ey:
            mat = compute_image(width, oh, xmin, xmax, sy, ey, state["max_iter"],
                               build_render_params(state))
            mat = mat[::-1, :, :]
            rgb = (mat * 255.0).astype(np.uint8)
            rgb_t = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
            strip = pygame.surfarray.make_surface(rgb_t)
            screen.blit(strip, (0, height + oy))


def run_render_mode(settings, cli_iter=None, cli_color=None, cli_gpu=False, cli_no_gpu=False):
    max_iter = cli_iter if cli_iter is not None else get_persistent_setting(settings, "iteration-max", cast=int, default=128)
    if cli_no_gpu:
        use_gpu = False
    else:
        use_gpu = cli_gpu or (settings.get("use-gpu", "false").lower() == "true")
    init_width = get_persistent_setting(settings, "width", cast=int, default=800)
    init_height = get_persistent_setting(settings, "height", cast=int, default=600)

    # Load keybinds from config, with defaults
    default_keybinds = {
        "quit": "escape",
        "toggle-info": "m",
        "reset-view": "o",
        "toggle-gpu": "p",
        "cycle-color": "c",
        "zoom-in": "scroll_up",
        "zoom-out": "scroll_down",
        "iter-up": "1",
        "iter-down": "2",
        "stripe-up": "3",
        "stripe-down": "4",
        "step-up": "5",
        "step-down": "6",
        "k-amb-up": "q",
        "k-amb-down": "w",
        "k-diff-up": "e",
        "k-diff-down": "r",
        "light-i-up": "a",
        "light-i-down": "s",
        "light-angle-up": "d",
        "light-angle-down": "f",
        "light-azim-up": "z",
        "light-azim-down": "x",
        "phase-up": "k",
        "phase-down": "l",
        "shininess-up": "n",
        "shininess-down": ",",
        "k-spec-up": "v",
        "k-spec-down": "b",
        "rgb-r-up": "7",
        "rgb-r-down": "8",
        "rgb-g-up": "9",
        "rgb-g-down": "0",
        "rgb-b-up": "y",
        "rgb-b-down": "h",
    }
    keybinds = {k: get_keybind(settings, k, v) for k, v in default_keybinds.items()}

    def key_name_to_pygame(name):
        """Convert keybind name to (pygame key constant, modifier) tuple.
        Supports formats like 'ctrl+r', 'shift+r', 'r'."""
        name = name.strip()
        mod = None
        if "+" in name:
            parts = name.split("+")
            mod = parts[0].strip().lower()
            name = parts[1].strip()
        kc = None
        if name.lower() == "escape":
            kc = pygame.K_ESCAPE
        elif name.lower() == "space":
            kc = pygame.K_SPACE
        elif name.lower() in ("enter", "return"):
            kc = pygame.K_RETURN
        elif name.lower() == "tab":
            kc = pygame.K_TAB
        elif name.lower() == "backspace":
            kc = pygame.K_BACKSPACE
        elif name.lower() in ("scroll_up", "scroll_down", "mouse1", "mouse2", "mouse3"):
            kc = None
        elif len(name) == 1:
            kc = pygame.key.key_code(name)
        else:
            kc = getattr(pygame, f"K_{name}", None)
        mod_map = {"ctrl": pygame.KMOD_CTRL, "shift": pygame.KMOD_SHIFT,
                   "alt": pygame.KMOD_ALT, "meta": pygame.KMOD_GUI}
        mod_val = mod_map.get(mod, 0) if mod else 0
        if mod:
            return (kc, mod_val)
        return kc

    def is_shift_held():
        """Check if shift modifier is currently held."""
        return bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

    def step_val(cur, delta, lo, hi, mult=1.0):
        """Step a value by delta*mult, clamped to [lo, hi]."""
        return round(max(lo, min(hi, cur + delta * mult)), 4)

    key_map = {}
    xmin, xmax = -2.5, 1.0
    ymin, ymax = -1.5, 1.5
    xmin, xmax, ymin, ymax = fix_aspect_ratio(xmin, xmax, ymin, ymax, init_width, init_height)

    pygame.init()
    screen = pygame.display.set_mode((init_width, init_height), pygame.RESIZABLE)
    pygame.display.set_caption("Mandelbrot — drag to pan, scroll to zoom, M for info")

    key_map = {}
    for action, name in keybinds.items():
        result = key_name_to_pygame(name)
        if result is not None:
            if isinstance(result, tuple):
                kc, mod_val = result
                key_map[kc] = (action, mod_val)
            else:
                key_map[result] = (action, 0)

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
            get_persistent_setting(settings, "rgb_r", cast=float, default=DEFAULT_RGB_THETAS[0]),
            get_persistent_setting(settings, "rgb_g", cast=float, default=DEFAULT_RGB_THETAS[1]),
            get_persistent_setting(settings, "rgb_b", cast=float, default=DEFAULT_RGB_THETAS[2]),
        ],
        "phase": get_persistent_setting(settings, "phase", cast=float, default=DEFAULT_PHASE),
        "use_gpu": use_gpu,
        "stripe_s": get_persistent_setting(settings, "stripe_s", cast=float, default=DEFAULT_STRIPE_S),
        "stripe_sig": get_persistent_setting(settings, "stripe_sig", cast=float, default=DEFAULT_STRIPE_SIG),
        "step_s": get_persistent_setting(settings, "step_s", cast=float, default=DEFAULT_STEP_S),
        "light_angle": get_persistent_setting(settings, "light_angle", cast=float, default=DEFAULT_LIGHT_ANGLE),
        "light_azim": get_persistent_setting(settings, "light_azim", cast=float, default=DEFAULT_LIGHT_AZIM),
        "light_i": get_persistent_setting(settings, "light_i", cast=float, default=DEFAULT_LIGHT_I),
        "k_ambiant": DEFAULT_K_AMBIANT,
        "k_diffuse": DEFAULT_K_DIFFUSE,
        "k_specular": DEFAULT_K_SPECULAR,
        "shininess": DEFAULT_SHININESS,
    }

    _t0 = time.perf_counter()
    compute_image(256, 256, xmin, xmax, ymin, ymax, 8,
                  build_render_params(state))
    pygame.event.pump()
    _cpu_warm_ms = (time.perf_counter() - _t0) * 1000
    print(f"[warmup] CPU JIT compiled in {_cpu_warm_ms:.0f} ms")
    _dbg(f"warmup backend=CPU {int(_cpu_warm_ms)}ms")

    if state["use_gpu"] and _CUDA_AVAILABLE:
        _t0 = time.perf_counter()
        try:
            compute_image(256, 256, xmin, xmax, ymin, ymax, 8,
                          build_render_params({**state, "use_gpu": True}))
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

    overlay = InfoOverlay()
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
                    _dbg(f"MOUSEBUTTONDOWN {event.pos}")
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
                    dragging = False
                    drag_offset_x = 0.0
                    drag_offset_y = 0.0
                    needs_render = True

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
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
                result = key_map.get(event.key)
                if result is None:
                    continue
                action, mod_val = result
                if mod_val != 0 and not bool(pygame.key.get_mods() & mod_val):
                    continue

                _mult = 10.0 if is_shift_held() else 1.0

                _num_steppers = {
                    "stripe-up":   lambda: state.update(stripe_s=(state["stripe_s"] + 1.0 * _mult) % 6.0),
                    "stripe-down": lambda: state.update(stripe_s=max(0.0, state["stripe_s"] - 1.0 * _mult)),
                    "step-up":     lambda: state.update(step_s=min(state["step_s"] + 1.0 * _mult, 8.0)),
                    "step-down":   lambda: state.update(step_s=max(0.0, state["step_s"] - 1.0 * _mult)),
                    "phase-up":    lambda: state.update(phase=step_val(state["phase"], 0.05, 0.0, 1.0, _mult)),
                    "phase-down":  lambda: state.update(phase=step_val(state["phase"], -0.05, 0.0, 1.0, _mult)),
                    "light-angle-up":   lambda: state.update(light_angle=step_val(state["light_angle"], 0.02, 0.0, 1.0, _mult)),
                    "light-angle-down": lambda: state.update(light_angle=step_val(state["light_angle"], -0.02, 0.0, 1.0, _mult)),
                    "light-azim-up":    lambda: state.update(light_azim=step_val(state["light_azim"], 0.02, 0.0, 1.0, _mult)),
                    "light-azim-down":  lambda: state.update(light_azim=step_val(state["light_azim"], -0.02, 0.0, 1.0, _mult)),
                    "light-i-up":       lambda: state.update(light_i=step_val(state["light_i"], 0.05, 0.0, 1.0, _mult)),
                    "light-i-down":     lambda: state.update(light_i=step_val(state["light_i"], -0.05, 0.0, 1.0, _mult)),
                    "k-amb-up":    lambda: state.update(k_ambiant=step_val(state["k_ambiant"], 0.05, 0.0, 1.0, _mult)),
                    "k-amb-down":  lambda: state.update(k_ambiant=step_val(state["k_ambiant"], -0.05, 0.0, 1.0, _mult)),
                    "k-diff-up":   lambda: state.update(k_diffuse=step_val(state["k_diffuse"], 0.05, 0.0, 1.0, _mult)),
                    "k-diff-down": lambda: state.update(k_diffuse=step_val(state["k_diffuse"], -0.05, 0.0, 1.0, _mult)),
                    "k-spec-up":   lambda: state.update(k_specular=step_val(state["k_specular"], 0.05, 0.0, 1.0, _mult)),
                    "k-spec-down": lambda: state.update(k_specular=step_val(state["k_specular"], -0.05, 0.0, 1.0, _mult)),
                    "shininess-up":   lambda: state.update(shininess=step_val(state["shininess"], 2.0, 0.0, 100.0, _mult)),
                    "shininess-down": lambda: state.update(shininess=step_val(state["shininess"], -2.0, 0.0, 100.0, _mult)),
                    "rgb-r-up":   lambda: state["rgb_thetas"].__setitem__(0, step_val(state["rgb_thetas"][0], 0.02, 0.0, 1.0, _mult)),
                    "rgb-r-down": lambda: state["rgb_thetas"].__setitem__(0, step_val(state["rgb_thetas"][0], -0.02, 0.0, 1.0, _mult)),
                    "rgb-g-up":   lambda: state["rgb_thetas"].__setitem__(1, step_val(state["rgb_thetas"][1], 0.02, 0.0, 1.0, _mult)),
                    "rgb-g-down": lambda: state["rgb_thetas"].__setitem__(1, step_val(state["rgb_thetas"][1], -0.02, 0.0, 1.0, _mult)),
                    "rgb-b-up":   lambda: state["rgb_thetas"].__setitem__(2, step_val(state["rgb_thetas"][2], 0.02, 0.0, 1.0, _mult)),
                    "rgb-b-down": lambda: state["rgb_thetas"].__setitem__(2, step_val(state["rgb_thetas"][2], -0.02, 0.0, 1.0, _mult)),
                }

                if action in _num_steppers:
                    _num_steppers[action]()
                    _RENDER_CACHE.clear()
                    needs_render = True

                elif action == "reset-view":
                    xmin, xmax = -2.5, 1.0
                    ymin, ymax = -1.5, 1.5
                    xmin, xmax, ymin, ymax = fix_aspect_ratio(xmin, xmax, ymin, ymax, width, height)
                    needs_render = True

                elif action == "iter-up":
                    state["max_iter"] = next_pow2(state["max_iter"])
                    needs_render = True

                elif action == "iter-down":
                    state["max_iter"] = max(1, prev_pow2(state["max_iter"]))
                    needs_render = True

                elif action == "cycle-color":
                    idx = 0
                    for i, preset in enumerate(COLOR_THETAS):
                        if [round(t, 2) for t in state["rgb_thetas"]] == [round(t, 2) for t in preset]:
                            idx = i
                            break
                    idx = (idx + 1) % len(COLOR_THETAS)
                    state["rgb_thetas"] = list(COLOR_THETAS[idx])
                    _RENDER_CACHE.clear()
                    needs_render = True

                elif action == "toggle-gpu":
                    state["use_gpu"] = not (state["use_gpu"] and _CUDA_AVAILABLE)
                    _RENDER_CACHE.clear()
                    needs_render = True

                elif action == "toggle-info":
                    overlay.toggle()
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

            if still_interacting or dragging:
                if surface is not None and (abs(drag_offset_x) > 0.1 or abs(drag_offset_y) > 0.1):
                    # Offset-blit existing surface + render only exposed edges
                    screen.fill((30, 30, 30))
                    screen.blit(surface, (int(drag_offset_x), int(drag_offset_y)))
                    _t0 = time.perf_counter()
                    _render_exposed_edges(screen, width, height, xmin, xmax,
                                          ymin, ymax, state, drag_offset_x,
                                          drag_offset_y)
                    _t_render = time.perf_counter() - _t0
                    needs_render = True
                else:
                    # No surface yet or negligible offset: full render
                    _t0 = time.perf_counter()
                    surface, _used_gpu = render_to_surface(
                        width, height, xmin, xmax, ymin, ymax, state["max_iter"], state)
                    _t_render = time.perf_counter() - _t0
                    last_render_xmin = xmin
                    last_render_xmax = xmax
                    last_render_ymin = ymin
                    last_render_ymax = ymax
                    needs_render = True
            else:
                interacting = False
                _t0 = time.perf_counter()
                surface, _used_gpu = render_to_surface(
                    width, height, xmin, xmax, ymin, ymax, state["max_iter"], state)
                _t_render = time.perf_counter() - _t0
                needs_render = False
                last_render_xmin = xmin
                last_render_xmax = xmax
                last_render_ymin = ymin
                last_render_ymax = ymax

        backend = "CUDA" if (state["use_gpu"] and _CUDA_AVAILABLE) else "CPU"
        _t_blit0 = time.perf_counter()
        if not needs_render:
            screen.blit(surface, (0, 0))
        elif not dragging:
            screen.blit(surface, (0, 0))
        elif abs(drag_offset_x) > 0.1 or abs(drag_offset_y) > 0.1:
            screen.blit(surface, (int(drag_offset_x), int(drag_offset_y)))
        else:
            screen.blit(surface, (0, 0))
        _t_blit = time.perf_counter() - _t_blit0
        if overlay.active:
            overlay.draw(screen, overlay_font, state, keybinds)
        _t_flip0 = time.perf_counter()
        pygame.display.flip()
        _t_flip = time.perf_counter() - _t_flip0
        _caption = f"Mandelbrot — iter={state['max_iter']} [{backend}]"
        if _caption != _last_caption:
            pygame.display.set_caption(_caption)
            _last_caption = _caption
        _dbg(f"frame={frame_num} abs={time.time():.3f} "
             f"total={(time.perf_counter()-_t_frame_start)*1000:.1f}ms "
             f"render={_t_render*1000:.1f}ms blit={_t_blit*1000:.1f}ms "
             f"flip={_t_flip*1000:.1f}ms "
             f"gpu={_used_gpu} hold={still_interacting} drag={dragging} size={width}x{height} iter={state['max_iter']} events={len(events)}")
        if frame_num % 30 == 0:
            print(f"[perf] frame={frame_num} total={time.perf_counter()-_t_frame_start:.1f}ms "
                  f"render={_t_render*1000:.0f}ms blit={_t_blit*1000:.0f}ms "
                  f"flip={_t_flip*1000:.0f}ms gpu={_used_gpu} drag={dragging} "
                  f"events={len(events)} iter={state['max_iter']} "
                  f"fps={frame_num / max(0.001, time.time() - _START_TIME):.1f}",
                  file=sys.stderr)
        frame_num += 1
        clock.tick(60)

    settings["width"] = str(width)
    settings["height"] = str(height)
    settings["iteration-max"] = str(state["max_iter"])
    settings["use-gpu"] = str(state["use_gpu"])
    settings["phase"] = str(state["phase"])
    settings["rgb_r"] = str(state["rgb_thetas"][0])
    settings["rgb_g"] = str(state["rgb_thetas"][1])
    settings["rgb_b"] = str(state["rgb_thetas"][2])
    settings["light_angle"] = str(state["light_angle"])
    settings["light_azim"] = str(state["light_azim"])
    settings["light_i"] = str(state["light_i"])
    settings["k_ambiant"] = str(state["k_ambiant"])
    settings["k_diffuse"] = str(state["k_diffuse"])
    settings["k_specular"] = str(state["k_specular"])
    settings["shininess"] = str(state["shininess"])
    settings["stripe_s"] = str(state["stripe_s"])
    settings["step_s"] = str(state["step_s"])
    for action, kc in keybinds.items():
        settings[f"keybind.{action}"] = kc
    print(f"[debug] use_gpu={state['use_gpu']} cuda_available={_CUDA_AVAILABLE} "
          f"final_size={width}x{height} iter={state['max_iter']} frames={frame_num}",
          file=sys.stderr)
    if persistent:
        save_settings(SETTINGS_FILE, settings)

    if _dbg_file is not None:
        _dbg("exit")
        _dbg_file.close()
    pygame.quit()


# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=None, help="Max iterations")
parser.add_argument("--color", type=str, default=None,
                    choices=["fire", "cool", "deep"], help="Color palette preset")
parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
parser.add_argument("--no-gpu", action="store_true", help="Force CPU rendering")
args = parser.parse_args()

# --- Mode selector ---
settings = load_settings(SETTINGS_FILE)
mode = input("Choose mode — (1) single point, (2) render image: ").strip()

if mode == "1":
    run_single_mode(settings)
elif mode == "2":
    run_render_mode(settings, cli_iter=args.iter, cli_color=args.color,
                    cli_gpu=args.gpu, cli_no_gpu=args.no_gpu)
else:
    print("Invalid mode selected.")
    sys.exit(1)
