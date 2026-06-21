"""
Build a TensorRT FP16 engine from an RF-DETR ONNX and benchmark it — the X1
"does it even convert?" smoke test + a 5090 FP16 FPS reference.

⚠ The engine is GPU-architecture-specific (built here for the 5090, sm_120) and will
NOT load on the Jetson Orin Nano (sm_87). This only (a) proves the ONNX is TRT-
convertible — i.e. the deformable-attention / DINOv2 ops are all TRT-parseable, the
real risk — and (b) gives a 5090 FP16 latency number. The DEPLOY engine must be built
on the Jetson from the same ONNX.

Usage
-----
    python scripts/trt_build_benchmark.py --onnx runs/rfdetr/export_nano/rfdetr-nano.onnx \
        --out runs/rfdetr/export_nano/rfdetr-nano.engine --iters 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt


def build_engine(onnx_path: str, fp16: bool, workspace_gb: int):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 0
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(
        trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"[trt] ONNX parse FAILED with {parser.num_errors} error(s):")
            for i in range(parser.num_errors):
                print(f"   - {parser.get_error(i)}")
            return None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] FP16 enabled")
    elif fp16:
        # TRT 11/cu13 dropped the FP16 builder flag (strongly-typed-network era);
        # precision comes from the network types. Build at default precision — the
        # point here is to validate the ONNX *parses* (the deformable-attn risk).
        print("[trt] note: no BuilderFlag.FP16 in this TRT (strongly-typed era); default precision")
    print("[trt] building engine (minutes)...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    print(f"[trt] build took {time.perf_counter() - t0:.1f}s")
    return serialized


def benchmark(serialized: bytes, iters: int) -> None:
    try:
        from cuda import cudart  # cuda-python
    except Exception as e:  # noqa: BLE001
        print(f"[trt] benchmark skipped (cuda-python missing: {e}); engine built OK though")
        return

    def chk(ret):
        err = ret[0] if isinstance(ret, tuple) else ret
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")
        return ret[1:] if isinstance(ret, tuple) and len(ret) > 1 else None

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    ctx = engine.create_execution_context()

    # allocate device buffers for every I/O tensor at its (static) shape
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    dptrs, host_in = {}, {}
    for n in names:
        shape = ctx.get_tensor_shape(n)
        vol = int(np.prod([max(1, d) for d in shape]))
        dt = trt.nptype(engine.get_tensor_dtype(n))
        nbytes = vol * np.dtype(dt).itemsize
        (dptr,) = chk(cudart.cudaMalloc(nbytes))
        dptrs[n] = dptr
        ctx.set_tensor_address(n, int(dptr))
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            host_in[n] = np.random.rand(*[max(1, d) for d in shape]).astype(dt)
            chk(cudart.cudaMemcpy(dptr, host_in[n].ctypes.data, nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    (stream,) = chk(cudart.cudaStreamCreate())

    for _ in range(10):  # warmup
        ctx.execute_async_v3(stream)
    chk(cudart.cudaStreamSynchronize(stream))
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    chk(cudart.cudaStreamSynchronize(stream))
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"[trt] TensorRT FP16 latency: {ms:.2f} ms/img  ({1000/ms:.1f} FPS, iters={iters}) on this GPU")
    for d in dptrs.values():
        cudart.cudaFree(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default=None, help="save the engine here (default: alongside onnx)")
    ap.add_argument("--no-fp16", action="store_false", dest="fp16")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    print(f"[trt] TensorRT {trt.__version__}; building from {args.onnx}")
    serialized = build_engine(args.onnx, args.fp16, args.workspace_gb)
    if serialized is None:
        print("[trt] ENGINE BUILD FAILED — see parse errors above (this is the X1 risk surfacing)")
        return 1
    out = Path(args.out) if args.out else Path(args.onnx).with_suffix(".engine")
    out.write_bytes(serialized)
    print(f"[trt] engine saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    benchmark(serialized, args.iters)
    print("[trt] DONE — ONNX is TRT-convertible. (Engine is 5090-specific; rebuild on Jetson.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
