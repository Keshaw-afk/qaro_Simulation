"""Time / energy / resource meter for every benchmark job. Appends one JSON line per measured block to
benchmark/results/measurements.jsonl (committed) so every number in RESULTS.md can be traced.

    with Meter("tabq_train_seed0", kind="train", method="tabq") as m:
        ...; m.add(env_steps=8_000_000)

What is recorded (null when not available on the machine):
- wall_s; cpu_user_s / cpu_sys_s of this process + finished children; peak RSS.
- CPU package energy (Intel RAPL, /sys/class/powercap/intel-rapl:0*/energy_uj; root-only by default on Fedora, see
  RESULTS.md). RAPL measures the WHOLE package, including other processes, so we also record the machine's total busy
  CPU time over the interval (/proc/stat) and an attributed share = package energy x (our CPU time / all busy CPU time).
- GPU energy (NVIDIA NVML total-energy counter, mJ) and mean/max GPU power and utilisation (sampled every second).
- Throughput (env_steps / wall_s when the job reports env_steps), mean CPU clock (MHz), max CPU temperature, and the power
  source (AC / battery) sampled every second: throttling or battery mode changes timings.
- Host description (CPU model, cores, GPU, Python/torch/numpy versions, git commit).
- cpu_energy_est_J: ESTIMATE used when RAPL is not readable (the owner chose not to open it, 2026-11-10):
  our CPU seconds x (CPU TDP / hardware threads). i5-12450H: 45 W / 12 threads = 3.75 W per busy thread; unknown CPUs
  (e.g. Colab's Xeon VM) 3.5 W. It ignores idle/uncore power and frequency changes, so it is a rough marginal-energy
  estimate (likely within a factor of ~2), not a measurement; the assumption is stored with every record."""
import json
import os
import pathlib
import platform
import resource
import subprocess
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "benchmark" / "results" / "measurements.jsonl"
RAPL = {"package": "/sys/class/powercap/intel-rapl:0", "core": "/sys/class/powercap/intel-rapl:0:0", "uncore": "/sys/class/powercap/intel-rapl:0:1"}


def _read(p):
    try:
        return int(open(p).read())
    except (OSError, ValueError):
        return None


def _rapl():
    return {k: _read(f"{d}/energy_uj") for k, d in RAPL.items()}


def _busy_cpu_s():
    """Total busy CPU seconds of the whole machine (all processes), from /proc/stat."""
    try:
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return (sum(f[:8]) - f[3] - f[4]) / os.sysconf("SC_CLK_TCK")  # user..steal minus idle and iowait (guest is already in user)
    except OSError:
        return None


def host_info():
    info = {"machine": platform.node(), "python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count()}
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    for mod in ("numpy", "torch", "stable_baselines3"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        info["git"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        info["git"] = None
    return info


def _cpu_mhz():
    try:
        f = [int(open(p).read()) for p in pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")]
        return sum(f) / len(f) / 1000 if f else None
    except OSError:
        return None


def _cpu_temp_c():
    try:
        for z in pathlib.Path("/sys/class/thermal").glob("thermal_zone*"):
            if (z / "type").read_text().strip() in ("x86_pkg_temp", "coretemp"):
                return int((z / "temp").read_text()) / 1000
        t = [int((z / "temp").read_text()) / 1000 for z in pathlib.Path("/sys/class/thermal").glob("thermal_zone*")]
        return max(t) if t else None
    except (OSError, ValueError):
        return None


def _mem_available_mb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) // 1024
    except OSError:
        pass
    return None


def _on_ac():
    for p in pathlib.Path("/sys/class/power_supply").glob("*/online"):
        try:
            return open(p).read().strip() == "1"
        except OSError:
            pass
    return None


class _GpuSampler(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_ev, self.h = period, [], threading.Event(), None
        self.cpu = []  # (MHz, temperature C, on AC, MemAvailable MB, 1-min load average)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def energy_mj(self):
        try:
            return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.h else None
        except Exception:
            return None

    def run(self):
        while not self.stop_ev.wait(self.period):
            self.cpu.append((_cpu_mhz(), _cpu_temp_c(), _on_ac(), _mem_available_mb(), os.getloadavg()[0]))
            if not self.h:
                continue
            try:
                self.samples.append((self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0, self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu,
                                     self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2 ** 20))
            except Exception:
                pass


class Meter:
    def __init__(self, name, log=LOG, **tags):
        self.name, self.log, self.tags, self.extra = name, pathlib.Path(log), tags, {}

    def add(self, **kw):
        self.extra.update(kw)

    def __enter__(self):
        self.gpu = _GpuSampler()
        self.gpu.start()
        self.t0, self.ru0, self.ruc0 = time.time(), resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        self.e0, self.busy0, self.g0 = _rapl(), _busy_cpu_s(), self.gpu.energy_mj()
        return self

    def __exit__(self, *exc):
        wall = time.time() - self.t0
        e1, busy1, g1 = _rapl(), _busy_cpu_s(), self.gpu.energy_mj()
        self.gpu.stop_ev.set()
        ru, ruc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_u = ru.ru_utime - self.ru0.ru_utime + ruc.ru_utime - self.ruc0.ru_utime
        cpu_s = ru.ru_stime - self.ru0.ru_stime + ruc.ru_stime - self.ruc0.ru_stime
        rec = {"name": self.name, **self.tags, "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)), "wall_s": round(wall, 3),
               "cpu_user_s": round(cpu_u, 3), "cpu_sys_s": round(cpu_s, 3), "peak_rss_mb": round(max(ru.ru_maxrss, ruc.ru_maxrss) / 1024, 1),
               "ok": exc[0] is None}
        for k in RAPL:
            if self.e0[k] is not None and e1[k] is not None:
                d = e1[k] - self.e0[k]
                if d < 0:  # counter wrapped
                    d += _read(f"{RAPL[k]}/max_energy_range_uj") or 0
                rec[f"cpu_{k}_J"] = round(d / 1e6, 2)
            else:
                rec[f"cpu_{k}_J"] = None
        if self.busy0 is not None and busy1 is not None:
            rec["machine_busy_cpu_s"] = round(busy1 - self.busy0, 2)
            share = (cpu_u + cpu_s) / max(busy1 - self.busy0, 1e-9)
            rec["cpu_share_of_machine"] = round(min(share, 1.0), 4)
            if rec["cpu_package_J"] is not None:
                rec["cpu_package_J_attributed"] = round(rec["cpu_package_J"] * min(share, 1.0), 2)
        rec["gpu_J"] = round((g1 - self.g0) / 1000, 2) if (self.g0 is not None and g1 is not None) else None
        if self.gpu.samples:
            p = [s[0] for s in self.gpu.samples]
            u = [s[1] for s in self.gpu.samples]
            rec.update(gpu_power_mean_W=round(sum(p) / len(p), 1), gpu_power_max_W=round(max(p), 1), gpu_util_mean_pct=round(sum(u) / len(u), 1),
                       gpu_mem_max_mb=round(max(s[2] for s in self.gpu.samples)), gpu_samples=len(p))
        mhz = [c[0] for c in self.gpu.cpu if c[0]]
        tmp = [c[1] for c in self.gpu.cpu if c[1]]
        ac = [c[2] for c in self.gpu.cpu if c[2] is not None]
        mem = [c[3] for c in self.gpu.cpu if c[3] is not None]
        load = [c[4] for c in self.gpu.cpu]
        rec.update(cpu_mhz_mean=round(sum(mhz) / len(mhz)) if mhz else None, cpu_temp_max_c=max(tmp) if tmp else None,
                   on_ac_all_time=all(ac) if ac else _on_ac(), mem_available_min_mb=min(mem) if mem else _mem_available_mb(),
                   loadavg_1min_mean=round(sum(load) / len(load), 2) if load else round(os.getloadavg()[0], 2),
                   end=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=" ".join(__import__("sys").argv), pid=os.getpid())
        model = host_info().get("cpu_model", "")
        w_thread, basis = (45.0 / 12, "i5-12450H TDP 45 W / 12 threads") if "i5-12450H" in model else (3.5, "generic 3.5 W per busy thread")
        rec.update(cpu_energy_est_J=round((cpu_u + cpu_s) * w_thread, 1), cpu_energy_est_basis=basis)
        rec.update(self.extra)
        if self.extra.get("env_steps"):
            rec["env_steps_per_s"] = round(self.extra["env_steps"] / max(wall, 1e-9), 1)
        for k in ("cpu_package_J", "cpu_package_J_attributed"):
            if rec.get(k) is not None:
                rec[k.replace("_J", "_W_mean")] = round(rec[k] / max(wall, 1e-9), 2)
        rec["host"] = host_info()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.record = rec
        return False
