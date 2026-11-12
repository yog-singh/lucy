"""Run `lucy voice` (file mode) as a subprocess; record stdout/stderr with times, the cold
start (launch to the banner's "Audio:" line), and the process's memory footprint and CPU.

usage (from the repo root): uv run python checks/voice_e2e/run_cli.py [--config FILE]
    [--label LABEL] [--out DIR] [--turns N] [--fast] [-- extra lucy voice args]
Plays inputs/t*.wav (listed in inputs/turns.json) as the user's turns, one after each
reply (--wait-for-reply), on a fresh database in OUT/LABEL/ (default
data/checks/voice_e2e/run/), where Lucy's audio (reply.wav), her per-turn timings
(voice_timings.jsonl) and this script's record (run.json) are written. Nothing is
played or recorded. Then run analyse.py with the same --label/--out.
See checks/README.md. macOS only (memory and CPU via proc_pid_rusage).
"""

import argparse
import ctypes
import ctypes.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_CONFIG, REPO, fresh_db, out_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(DEFAULT_CONFIG))
ap.add_argument("--label", default="run")
ap.add_argument("--out", help="output directory (default: data/checks/voice_e2e/)")
ap.add_argument("--turns", type=int, help="play only the first N turns (smoke run)")
ap.add_argument(
    "--fast", action="store_true", help="simulated time: quick, but latencies are not real"
)
ap.add_argument("extra", nargs="*", help="after --: more `lucy voice` arguments")
args = ap.parse_args()
name = args.label
run_dir = out_dir("voice_e2e", args.out) / name
db = fresh_db(run_dir)
turns = json.loads((HERE / "inputs" / "turns.json").read_text())[: args.turns or None]
argv = [
    "voice",
    "--config",
    str(Path(args.config).resolve()),
    "--session",
    f"check-{name}",
    "--input-wav",
    *[str(HERE / "inputs" / t["file"]) for t in turns],
    "--output-wav",
    str(run_dir / "reply.wav"),
    "--wait-for-reply",
    *(["--fast"] if args.fast else []),
    *args.extra,
]


class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "user",
            "system",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired",
            "resident",
            "phys_footprint",
            "start",
            "exit",
            "cu",
            "cs",
            "cw",
            "ciw",
            "cp",
            "ce",
            "dr",
            "dw",
        )
    ]


libc = ctypes.CDLL(ctypes.util.find_library("c"))


def usage(pid):
    ru = RU()
    if libc.proc_pid_rusage(pid, 2, ctypes.byref(ru)) != 0:
        return None
    return ru.phys_footprint / 2**20, (
        ru.user + ru.system
    ) / 1e9  # Mach abstime is ns on Apple silicon


t0 = time.monotonic()
# `lucy ARGV` with its database redirected to `db` (see checks/common.py)
proc = subprocess.Popen(
    [sys.executable, str(HERE.parent / "common.py"), "lucy", str(db), "--", *argv],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    cwd=REPO,
)
lines = []


def reader(stream, tag):
    for line in stream:
        lines.append((round(time.monotonic() - t0, 3), tag, line.rstrip("\n")))


for s, tag in ((proc.stdout, "out"), (proc.stderr, "err")):
    threading.Thread(target=reader, args=(s, tag), daemon=True).start()
samples = []
while proc.poll() is None:
    u = usage(proc.pid)
    if u:
        samples.append((round(time.monotonic() - t0, 2), round(u[0], 1), round(u[1], 3)))
    time.sleep(0.5)
time.sleep(0.2)
ready = next((t for t, tag, l in lines if tag == "err" and l.startswith("Audio:")), None)
result = {
    "name": name,
    "argv": argv,
    "turns": len(turns),
    "exit": proc.returncode,
    "seconds": round(time.monotonic() - t0, 1),
    "cold_start_s": ready,
    "lines": lines,
    "samples": samples,
    "peak_footprint_mb": max((s[1] for s in samples), default=None),
}
json.dump(result, open(run_dir / "run.json", "w"), indent=1)
for t, tag, l in lines:
    print(f"{t:7.2f} {tag} {l}")
print("exit", proc.returncode, "cold start", ready, "peak MB", result["peak_footprint_mb"])
print("written to", run_dir)
