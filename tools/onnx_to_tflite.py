#!/usr/bin/env python3
"""Convert an ONNX model to TFLite (ONNX -> TF SavedModel -> TFLite via onnx2tf) and check it
against onnxruntime.

Setup (Python 3.10/3.11, offline tool, not part of the Gradle build):
    pip install onnx onnxruntime onnx2tf tensorflow tf_keras onnxsim onnx_graphsurgeon sng4onnx \
                ai_edge_litert psutil

Examples:
    # FireRedVAD: feat [1,T,80]; T is dynamic, pin it for the conversion (the app resizes at runtime)
    python tools/onnx_to_tflite.py app/src/main/assets/models/fireredvad_vad.onnx \
        --shape feat=1,100,80 --verify

    # FireRedVAD AED (same input, probs [1,T,3])
    python tools/onnx_to_tflite.py app/src/main/assets/models/fireredvad_aed.onnx \
        --shape feat=1,100,80 --verify

    # Float16 weights (about half the size), output next to the input
    python tools/onnx_to_tflite.py model.onnx --fp16 --verify

Notes:
  * Dynamic dims must be fixed with --shape NAME=d0,d1,... (repeatable). The app's TFLite path
    resizes input tensors, so a short fixed T is fine as long as the converted graph has no
    shape-dependent constants; --verify tests two different lengths for that reason
    (--verify-lens, only for inputs that have a dynamic dim).
  * The app matches tensors by name and falls back to positional order (see inference.cpp), so
    keep the input/output order of the original model.
  * onnx2tf renames tensors and may transpose NCHW -> NHWC for 4-D tensors; --verify compares
    outputs positionally and reports max abs error. Anything above --tol means the conversion is
    not faithful: do NOT ship it, and compare layer by layer with `onnx2tf -cotof`.
  * After converting, check the result with NativeVadTest against the Python reference
    (CLAUDE.md): fbank/postprocess are ports, the model file must match too.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_shapes(items):
    shapes = {}
    for it in items or []:
        name, _, dims = it.partition("=")
        if not dims:
            sys.exit(f"--shape expects NAME=d0,d1,...  got: {it}")
        shapes[name] = [int(d) for d in dims.split(",")]
    return shapes


def convert(src: Path, out_dir: Path, shapes, fp16: bool, extra):
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(src), "-o", str(out_dir), "-osd", "-dgc"]
    # -osd: also write a SavedModel; -dgc: disable group convolution rewrite (smaller, safer)
    for name, dims in shapes.items():
        cmd += ["-ois", f"{name}:{','.join(map(str, dims))}"]
    cmd += extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    pick = "model_float16.tflite" if fp16 else "model_float32.tflite"
    stem = src.stem
    for cand in (out_dir / f"{stem}_{pick.split('model_')[1]}", out_dir / pick):
        if cand.exists():
            return cand
    found = sorted(out_dir.glob("*.tflite"))
    if not found:
        sys.exit("onnx2tf produced no .tflite file")
    wanted = [p for p in found if ("float16" in p.name) == fp16]
    return (wanted or found)[0]


def _interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(path))


def verify(onnx_path: Path, tflite_path: Path, shapes, lens, tol: float, seed: int):
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    ok = True

    def make(dyn_len):
        feeds = {}
        for i in ins:
            fixed = shapes.get(i.name)
            dims = []
            for k, od in enumerate(i.shape):
                if isinstance(od, int):
                    dims.append(od)
                elif dyn_len is not None and not (fixed and k == 0):
                    dims.append(dyn_len)  # dynamic axis (batch axis 0 stays as pinned)
                else:
                    dims.append(fixed[k] if fixed else 1)
            if i.type == "tensor(float)":
                feeds[i.name] = rng.standard_normal(dims).astype(np.float32)
            elif i.type == "tensor(int64)":
                feeds[i.name] = np.ones(dims, np.int64)
            else:
                sys.exit(f"unsupported input type {i.type} for {i.name}")
        return feeds

    has_dyn = any(not isinstance(d, int) for i in ins for d in i.shape)
    runs = lens if has_dyn else [None]
    for n in runs:
        feeds = make(n)
        ref = sess.run(None, feeds)
        it = _interp(tflite_path)
        det = it.get_input_details()
        if len(det) != len(ins):
            print(f"  input count differs: onnx={len(ins)} tflite={len(det)}")
            ok = False
            continue
        # positional match; tflite input order follows the ONNX order in onnx2tf
        for d, (name, arr) in zip(det, feeds.items()):
            if list(d["shape"]) != list(arr.shape):
                it.resize_tensor_input(d["index"], list(arr.shape), strict=False)
        it.allocate_tensors()
        for d, (name, arr) in zip(it.get_input_details(), feeds.items()):
            it.set_tensor(d["index"], arr.astype(d["dtype"]))
        it.invoke()
        outs = [it.get_tensor(o["index"]) for o in it.get_output_details()]
        tag = f"T={n}" if n is not None else "fixed"
        if len(outs) != len(ref):
            print(f"  [{tag}] output count differs: onnx={len(ref)} tflite={len(outs)}")
            ok = False
            continue
        for k, (a, b) in enumerate(zip(ref, outs)):
            if a.shape != b.shape:
                # NHWC/NCHW or squeezed dims: compare after dropping size-1 axes
                if np.squeeze(a).shape == np.squeeze(b).shape:
                    a, b = np.squeeze(a), np.squeeze(b)
                else:
                    print(f"  [{tag}] output {k}: shape onnx={a.shape} tflite={b.shape}  MISMATCH")
                    ok = False
                    continue
            err = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            good = err <= tol
            ok &= good
            print(f"  [{tag}] output {k} {a.shape}: max|diff|={err:.3e}  {'OK' if good else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output .tflite (default: <onnx stem>.tflite next to the input)")
    ap.add_argument("--shape", action="append", metavar="NAME=D0,D1,..",
                    help="fix an input's shape for conversion (repeatable)")
    ap.add_argument("--fp16", action="store_true", help="use the float16 variant")
    ap.add_argument("--verify", action="store_true", help="compare with onnxruntime on random inputs")
    ap.add_argument("--verify-lens", type=int, nargs="+", default=[50, 137],
                    help="lengths of the dynamic axis tried by --verify (default: 50 137)")
    ap.add_argument("--tol", type=float, default=None,
                    help="max abs error allowed by --verify (default 1e-3, or 2e-2 with --fp16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=Path, help="keep onnx2tf's working output (SavedModel etc.) in this folder")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra onnx2tf args after `--`")
    a = ap.parse_args()

    src = a.onnx.resolve()
    if not src.exists():
        sys.exit(f"not found: {src}")
    dst = (a.output or src.with_suffix(".tflite")).resolve()
    shapes = parse_shapes(a.shape)
    extra = [x for x in a.extra if x != "--"]
    tol = a.tol if a.tol is not None else (2e-2 if a.fp16 else 1e-3)

    work = a.keep.resolve() if a.keep else Path(tempfile.mkdtemp(prefix="onnx2tf_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        produced = convert(src, work, shapes, a.fp16, extra)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, dst)
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.2f} MB, onnx {src.stat().st_size / 1e6:.2f} MB)")
        if a.verify:
            print("verify vs onnxruntime:")
            if not verify(src, dst, shapes, a.verify_lens, tol, a.seed):
                sys.exit("verification FAILED, do not ship this model")
            print("verification OK")
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
