#!/usr/bin/env python3
"""Sample GPU and host memory every --interval seconds into a CSV until killed.

GB10 has unified CPU/GPU memory: nvidia-smi often reports memory.used as N/A, so host
MemAvailable from /proc/meminfo is recorded too (that is the real memory pressure signal).
    python scripts/telemetry.py --output results/telemetry.csv --interval 2
"""
import argparse
import csv
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["utilization.gpu", "memory.used", "memory.total", "power.draw", "temperature.gpu", "clocks.sm"]


def query(fields):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip().splitlines()[0].split(", ") if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def meminfo():
    vals = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                vals[k] = int(v.split()[0]) // 1024  # MiB
    except OSError:
        pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until killed")
    a = ap.parse_args()
    fields = [f for f in FIELDS if query([f]) is not None]  # one unsupported field breaks the whole query
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    t0 = time.monotonic()
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "elapsed_s"] + fields + ["host_mem_total_mib", "host_mem_available_mib",
                                                              "host_mem_used_mib"])
        try:
            while not stop and (not a.duration or time.monotonic() - t0 < a.duration):
                vals = query(fields) if fields else []
                m = meminfo()
                used = m["MemTotal"] - m["MemAvailable"] if len(m) == 2 else ""
                w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            round(time.monotonic() - t0, 1)] + (vals or [""] * len(fields))
                           + [m.get("MemTotal", ""), m.get("MemAvailable", ""), used])
                f.flush()
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass
    print(f"telemetry saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
