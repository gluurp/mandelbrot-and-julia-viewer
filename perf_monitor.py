#!/usr/bin/env python3
import time
import threading
import subprocess
import json
import os
import sys
import queue

class PerfMonitor:
    """Background process and GPU/CPU performance monitor."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue(maxsize=2)
        self.stats = {"fps": 0, "cpu": 0.0, "gpu_util": 0.0, "gpu_mem_used": 0, "gpu_mem_total": 0,
                      "mem_used": 0, "mem_total": 0}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_stats(self):
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return self.stats

    def _query_nvidia_smi(self, stats):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    stats["gpu_util"] = float(parts[0])
                    stats["gpu_mem_used"] = float(parts[1])  # already in MB
                    stats["gpu_mem_total"] = float(parts[2])  # already in MB
        except Exception:
            pass

    def _loop(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        nvml = False
        gpu_info = {}

        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
            nvmlInit()
            nvml = True
        except Exception:
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpu_info["model"] = result.stdout.strip()
            except Exception:
                pass

        last_time = time.perf_counter()
        last_fps_time = time.perf_counter()
        frame_count = 0

        while self.running:
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now

            stats = dict(self.stats)

            # CPU usage
            if psutil:
                stats["cpu"] = psutil.cpu_percent(interval=None)
                # Memory
                vm = psutil.virtual_memory()
                stats["mem_used"] = vm.used / (1024**3)
                stats["mem_total"] = vm.total / (1024**3)

            # GPU stats
            if nvml:
                try:
                    handle = nvmlDeviceGetHandleByIndex(0)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    stats["gpu_util"] = util.gpu
                    stats["gpu_mem_used"] = mem_info.used / (1024**2)
                    stats["gpu_mem_total"] = mem_info.total / (1024**2)
                except Exception:
                    pass
            else:
                self._query_nvidia_smi(stats)

            # FPS (updated by external counter)
            stats["fps"] = frame_count / max(elapsed, 0.001)

            # Push to queue (drop old if full)
            try:
                self.data_queue.put_nowait(stats)
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(stats)
                except Exception:
                    pass

            self.stats = dict(stats)
            time.sleep(self.interval)


class RenderProfiler:
    """Profiles render timing and tile distribution."""

    def __init__(self):
        self.render_times = []
        self.tile_counts = {"gpu_tiles": 0, "cpu_tiles": 0}
        self.max_tile_counts = {"gpu_tiles": 0, "cpu_tiles": 0}
        self.max_history = 60

    def record_render(self, elapsed_ms, n_gpu_tiles, n_cpu_tiles):
        self.render_times.append(elapsed_ms)
        if len(self.render_times) > self.max_history:
            self.render_times.pop(0)
        self.tile_counts = {"gpu_tiles": n_gpu_tiles, "cpu_tiles": n_cpu_tiles}
        if n_gpu_tiles > self.max_tile_counts["gpu_tiles"] or n_cpu_tiles > self.max_tile_counts["cpu_tiles"]:
            if n_gpu_tiles > self.max_tile_counts["gpu_tiles"]:
                self.max_tile_counts["gpu_tiles"] = n_gpu_tiles
            if n_cpu_tiles > self.max_tile_counts["cpu_tiles"]:
                self.max_tile_counts["cpu_tiles"] = n_cpu_tiles

    def get_avg_render_ms(self):
        return sum(self.render_times) / len(self.render_times) if self.render_times else 0

    def get_stats(self):
        return {
            "avg_render_ms": self.get_avg_render_ms(),
            "tile_counts": self.tile_counts,
            "max_tile_counts": self.max_tile_counts,
            "tile_distribution": "GPU:{} CPU:{} (max: GPU:{} CPU:{})".format(
                self.tile_counts["gpu_tiles"], self.tile_counts["cpu_tiles"],
                 self.max_tile_counts["gpu_tiles"], self.max_tile_counts["cpu_tiles"])
        }

