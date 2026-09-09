#!/usr/bin/env python3
"""
Launch mandelbrot viewer and monitor performance in real-time.
Displays FPS, render times, CPU/GPU utilization.
"""
import subprocess
import time
import threading
import queue
import re
import sys
import os

try:
    from perf_monitor import PerfMonitor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from perf_monitor import PerfMonitor


def parse_perf_line(line):
    """Parse [perf] lines from stderr."""
    if not line.startswith("[perf]"):
        return None
    result = {"raw": line.strip()}
    patterns = [
        (r"frame=(\d+)", "frame"),
        (r"total=([\d.]+)ms", "total_ms"),
        (r"render=([\d.]+)ms", "render_ms"),
        (r"blit=([\d.]+)ms", "blit_ms"),
        (r"flip=([\d.]+)ms", "flip_ms"),
        (r"gpu=(True|False)", "gpu"),
        (r"fps=([\d.]+)", "fps"),
        (r"avg_render=([\d.]+)ms", "avg_render_ms"),
        (r"tiles=([\d.]+)/([\d.]+)", "gpu_tiles", "cpu_tiles"),
        (r"sys_cpu=([\d.]+)%", "sys_cpu"),
        (r"gpu_util=([\d.]+)%", "gpu_util"),
    ]
    for pattern, *keys in patterns:
        m = re.search(pattern, line)
        if m:
            if len(keys) == 1:
                result[keys[0]] = m.group(1)
            else:
                for i, k in enumerate(keys):
                    result[k] = m.group(i + 1)
    return result


class PerfLauncher:
    def __init__(self):
        self.monitor = PerfMonitor(interval=0.5)
        self.running = False
        self.proc = None
        self.stats_history = []
        self.max_history = 60

    def start(self):
        self.monitor.start()
        self.running = True

        # Start the app
        env = os.environ.copy()
        env["MB_DEBUG_PERF"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandelbrot_testing_playground.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # Send '2' to select render mode
        self.proc.stdin.write("2\n")
        self.proc.stdin.flush()

        # Start reader threads
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.monitor.stop()

    def _read_stdout(self):
        while self.running and self.proc:
            line = self.proc.stdout.readline()
            if not line:
                break
            # Could parse pygame window events here

    def _read_stderr(self):
        while self.running and self.proc:
            line = self.proc.stderr.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("[perf]"):
                parsed = parse_perf_line(line)
                if parsed:
                    self.stats_history.append(parsed)
                    if len(self.stats_history) > self.max_history:
                        self.stats_history.pop(0)

    def get_summary(self):
        if not self.stats_history:
            return "No perf data yet"
        latest = self.stats_history[-1]
        mon = self.monitor.get_stats()
        parts = [
            f"FPS: {latest.get('fps', '?')}",
            f"Render: {latest.get('render_ms', '?')}ms",
            f"CPU: {mon.get('cpu', '?')}%",
            f"GPU util: {mon.get('gpu_util', '?')}%",
        ]
        if "gpu_tiles" in latest:
            parts.append(f"Tiles: GPU:{latest['gpu_tiles']} CPU:{latest['cpu_tiles']}")
        return " | ".join(parts)

    def print_stats(self):
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        if not self.stats_history:
            print("No perf data collected")
            return

        # Latest stats
        latest = self.stats_history[-1]
        print(f"\nLatest frame:")
        print(f"  FPS: {latest.get('fps', '?')}")
        print(f"  Render: {latest.get('render_ms', '?')}ms")
        print(f"  Blit: {latest.get('blit_ms', '?')}ms")
        print(f"  Flip: {latest.get('flip_ms', '?')}ms")
        print(f"  GPU: {latest.get('gpu', '?')}")

        # System stats
        mon = self.monitor.get_stats()
        print(f"\nSystem:")
        print(f"  CPU: {mon.get('cpu', '?')}%")
        print(f"  Memory: {mon.get('mem_used', '?'):.1f} / {mon.get('mem_total', '?'):.1f} GB")
        print(f"  GPU util: {mon.get('gpu_util', '?')}%")
        print(f"  GPU mem: {mon.get('gpu_mem_used', '?'):.0f} / {mon.get('gpu_mem_total', '?'):.1f} MB")

        # Tile distribution
        if "gpu_tiles" in latest:
            print(f"\nTiles: GPU:{latest['gpu_tiles']} CPU:{latest['cpu_tiles']}")

        # Averages
        if len(self.stats_history) > 1:
            fps_vals = [float(s.get("fps", 0)) for s in self.stats_history if s.get("fps")]
            render_vals = [float(s.get("render_ms", 0)) for s in self.stats_history if s.get("render_ms")]
            if fps_vals:
                print(f"\nAverages (last {len(fps_vals)} frames):")
                print(f"  FPS: {sum(fps_vals)/len(fps_vals):.1f}")
            if render_vals:
                print(f"  Render: {sum(render_vals)/len(render_vals):.1f}ms")

        print("=" * 60)


def main():
    launcher = PerfLauncher()
    try:
        print("Starting Mandelbrot viewer with performance monitoring...")
        print("Press Ctrl+C to stop and see summary")
        launcher.start()
        while True:
            time.sleep(2)
            if launcher.stats_history:
                print(f"\r{launcher.get_summary()}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        launcher.stop()
        launcher.print_stats()


if __name__ == "__main__":
    main()
