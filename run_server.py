"""
run_server.py — self-healing supervisor for the NeuroSense API
================================================================
This host's NVIDIA driver has a confirmed bug: something in this
process briefly touches a GPU/Direct3D context, and an NVIDIA driver
background thread later access-violates inside nvdxgdmal64.dll after
that context is torn down (Windows Event Viewer shows the identical
fault — same module, same offset 0x86f0 — regardless of which Python
library or code path triggers it). Every application-level mitigation
that's actually under this codebase's control has been applied
(MediaPipe forced to CPU delegate, OpenCV hardware video decode and
OpenCL both disabled, ANGLE forced to swiftshader, ffmpeg's validation
subprocess bypassed, torch capped to CPU/4 threads) and the crash still
recurs — this is a bug in the installed driver itself, not fixable
from Python.

Since the crash is unpredictable but rare and the process otherwise
runs correctly, this supervisor makes the API self-healing: it runs
uvicorn as a child process and immediately restarts it if it ever
dies unexpectedly, so an access violation costs a few seconds of
downtime (plus whatever in-flight requests were lost) instead of
leaving the API down until someone notices and restarts it by hand.

Run this instead of `python -m uvicorn main:app ...` directly:
    python run_server.py
"""

import subprocess
import sys
import time

HOST = "0.0.0.0"
PORT = "8000"

# If uvicorn dies faster than this after starting, treat it as a
# startup failure (bad code, missing dependency, etc.) rather than the
# driver crash, and stop retrying instead of spinning forever.
MIN_HEALTHY_SECONDS = 15
MAX_CONSECUTIVE_FAST_FAILURES = 5


def main():
    fast_failures = 0
    restart_count = 0

    while True:
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", HOST, "--port", PORT, "--workers", "1",
        ]
        print(f"[supervisor] starting uvicorn (restart #{restart_count}) ...", flush=True)
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd)
        exit_code = proc.wait()
        uptime = time.monotonic() - t0

        print(f"[supervisor] uvicorn exited with code {exit_code} after {uptime:.1f}s", flush=True)

        if exit_code == 0:
            # Clean shutdown (e.g. Ctrl+C propagated to the child) — don't restart.
            print("[supervisor] clean exit, not restarting.", flush=True)
            break

        if uptime < MIN_HEALTHY_SECONDS:
            fast_failures += 1
            if fast_failures >= MAX_CONSECUTIVE_FAST_FAILURES:
                print(
                    f"[supervisor] {fast_failures} crashes within "
                    f"{MIN_HEALTHY_SECONDS}s of startup in a row — "
                    "this looks like a real bug, not the known driver "
                    "crash. Giving up; check the traceback above.",
                    flush=True,
                )
                break
        else:
            fast_failures = 0

        restart_count += 1
        print("[supervisor] restarting in 2s ...", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
