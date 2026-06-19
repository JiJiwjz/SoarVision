"""
Multi-GPU experiment launcher — one experiment per GPU (NOT DDP).

YOLO26n is tiny (2.7M params); DDP on a single run scales poorly. The throughput
win comes from running the ablation grid in parallel: each GPU pulls the next
experiment from a shared queue and runs train.py on it serially. Heterogeneous
GPUs are fine (e.g. mixing 4090 + 5090) — slower cards just finish fewer jobs.

Each child sees only its assigned GPU via CUDA_VISIBLE_DEVICES, so train.py is
always called with --device 0.

Usage
-----
    python scripts/run_grid.py --gpus 0,1,2,3 --grid demo
    python scripts/run_grid.py --gpus 0,1 --grid d1 --epochs 100
    python scripts/run_grid.py --gpus 0,1,2,3 --grid full --dry-run
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = REPO_ROOT / "scripts" / "train.py"
LOG_DIR = REPO_ROOT / "runs" / "grid_logs"


def build_grid(name: str, epochs: int, imgsz_list: list[int]) -> list[dict]:
    """Each experiment is a dict of train.py flags (without --device)."""
    stages = {
        "demo": [("e0", False), ("e1", False), ("e2", False)],
        "d1":   [("e1", False), ("e1", True), ("e2", False), ("e2", True)],
        "full": [(s, d) for s in ("e0", "e1", "e2") for d in (False, True)],
    }[name]
    exps = []
    for stage, d1 in stages:
        for imgsz in imgsz_list:
            exp = {"--stage": stage, "--epochs": str(epochs), "--imgsz": str(imgsz)}
            tag = stage
            if d1:
                exp["--d1"] = True
                tag += "_d1"
            if len(imgsz_list) > 1:
                tag += f"_{imgsz}"
            exp["--name"] = f"{tag}_26n"
            exps.append(exp)
    # Enqueue longest jobs first so high-imgsz runs claim GPUs early and don't
    # become end-of-grid stragglers on a small GPU pool.
    exps.sort(key=lambda e: -int(e["--imgsz"]))
    return exps


def exp_to_cmd(exp: dict, python: str) -> list[str]:
    cmd = [python, str(TRAIN), "--device", "0"]
    for k, v in exp.items():
        if v is True:
            cmd.append(k)
        else:
            cmd += [k, str(v)]
    return cmd


def worker(gpu: str, q: "queue.Queue[dict]", python: str, dry_run: bool, lock: threading.Lock):
    while True:
        try:
            exp = q.get_nowait()
        except queue.Empty:
            return
        name = exp.get("--name", "exp")
        cmd = exp_to_cmd(exp, python)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with lock:
            print(f"[gpu {gpu}] START {name}: {' '.join(cmd)}")
        if dry_run:
            q.task_done()
            continue
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / f"{name}_gpu{gpu}.log"
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        with lock:
            mins = (time.time() - t0) / 60
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[gpu {gpu}] DONE  {name} {status} in {mins:.1f}m → {log}")
        q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU ids, e.g. 0,1,2,3")
    ap.add_argument("--grid", default="demo", choices=["demo", "d1", "full"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", default="960", help="comma list, e.g. 960 or 640,960,1280")
    ap.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    # Hardware passthrough → appended to every experiment's train.py call.
    # On a shared node (N parallel runs) keep workers ≈ cpu_cores / N and avoid
    # cache=ram unless one run fits the host RAM N times over.
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None, help="disk | ram | False")
    ap.add_argument("--fraction", type=float, default=None,
                    help="train on a subset, e.g. 0.3 — for fast preview runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    imgsz_list = [int(x) for x in str(args.imgsz).split(",")]
    exps = build_grid(args.grid, args.epochs, imgsz_list)
    for e in exps:
        if args.batch is not None:
            e["--batch"] = str(args.batch)
        if args.workers is not None:
            e["--workers"] = str(args.workers)
        if args.cache is not None:
            e["--cache"] = args.cache
        if args.fraction is not None:
            e["--fraction"] = str(args.fraction)

    print(f"[grid] {len(exps)} experiments across {len(gpus)} GPU(s): {gpus}")
    for e in exps:
        print("   -", e.get("--name"), {k: v for k, v in e.items() if k != "--name"})

    q: "queue.Queue[dict]" = queue.Queue()
    for e in exps:
        q.put(e)

    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, q, args.python, args.dry_run, lock))
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("[grid] all experiments finished" if not args.dry_run else "[grid] dry-run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
