"""Export FireRedVAD (VAD or AED) with a fixed time axis to ONNX and/or TFLite, for NPU
runtimes (Samsung SNAP).

Offline tool, not part of the build. Needs torch and the FireRedVAD sources: either a checkout
(--repo, only fireredvad/core/detect_model.py is used) or the installed `fireredvad` package,
plus model.pth.tar from https://huggingface.co/FireRedTeam/FireRedVAD. ONNX output needs `onnx`;
TFLite output needs only the pure-Python `tflite` and `flatbuffers` packages (no TensorFlow):

    pip install torch onnx tflite flatbuffers
    python tools/export_firered_fixed.py --ckpt VAD/model.pth.tar --frames 1000 \
        --out fireredvad_vad_t1000.tflite [fireredvad_vad_t1000.onnx] [--channel-dim]

I/O (all dimensions static; the format follows each --out extension):
    feat  [1, T, 80]  (or [1, T, 80, 1] with --channel-dim)  fbank + CMVN
    probs [1, T, odim]  sigmoid scores (odim = 1 for VAD; speech, singing, music for AED)

The TFLite graph is written directly from the PyTorch weights, kept in an NHWC [1, 1, T, C]
layout that NPUs handle well: every Linear is a 1x1 CONV_2D, and every FSMN layer (residual +
lookback + lookahead filters) is a single PAD + DEPTHWISE_CONV_2D whose taps span t-19..t+20,
with the identity folded into the centre tap. Only CONV_2D, DEPTHWISE_CONV_2D, PAD, ADD,
LOGISTIC and RESHAPE are used. If ai-edge-litert can be imported, the file is checked against
PyTorch.

The app (FireRedVadModel::RunFixed) slides overlapping T-frame windows over each chunk and
keeps only frames that see the model's whole receptive field, so T must exceed lookback +
lookahead (printed below; 312 frames for the released models) for results to equal the
dynamic model. kLookback / kLookahead in vad_models.h must match the printed values.

The checkpoint is loaded with weights_only=True (only argparse.Namespace is allowed),
unlike DetectModel.from_pretrained, so a tampered file cannot run code.
"""

import argparse
import importlib
import importlib.util
import os

import numpy as np
import torch


def load_detect_model_class(repo):
    # Load the module by path: importing the fireredvad package pulls in audio deps.
    if repo:
        path = os.path.join(repo, "fireredvad", "core", "detect_model.py")
    else:
        spec = importlib.util.find_spec("fireredvad")
        if spec is None:
            raise SystemExit("fireredvad is not installed; pass --repo")
        path = os.path.join(os.path.dirname(spec.origin), "core", "detect_model.py")
    spec = importlib.util.spec_from_file_location("detect_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DetectModel


class NonStreaming(torch.nn.Module):
    def __init__(self, model, frames, channel_dim):
        super().__init__()
        self.model = model
        self.frames = frames
        self.channel_dim = channel_dim

    def forward(self, feat):
        if self.channel_dim:
            feat = feat.reshape(1, self.frames, 80)
        probs, _ = self.model(feat, caches=None)
        return probs


def fsmn_kernel(fsmn):
    """Folds an FSMN layer into one depthwise kernel [taps, P] over offsets -left..+right."""
    n1, s1, n2, s2 = fsmn.N1, fsmn.S1, fsmn.N2, fsmn.S2
    left, right = (n1 - 1) * s1, n2 * s2
    back = fsmn.lookback_filter.weight.detach().numpy()[:, 0, :]  # [P, N1]
    kernel = np.zeros((left + 1 + right, back.shape[0]), np.float32)
    for j in range(n1):  # lookback tap j reads frame t - (N1-1-j)*S1
        kernel[j * s1] += back[:, j]
    if n2 > 0:  # lookahead tap j reads frame t + (j+1)*S2 (FSMN.forward slices off N2*S2)
        ahead = fsmn.lookahead_filter.weight.detach().numpy()[:, 0, :]
        for j in range(n2):
            kernel[left + (j + 1) * s2] += ahead[:, j]
    kernel[left] += 1.0  # FSMN residual
    return kernel, left, right


class TfliteWriter:
    """Builds a single-subgraph float32 TFLite flatbuffer."""

    def __init__(self):
        import flatbuffers

        self.fb = flatbuffers
        self.tensors = []  # (name, shape, dtype, buffer index)
        self.buffers = [None]  # buffer 0 is the empty sentinel
        self.ops = []  # (builtin code, inputs, outputs, options type, options builder)
        self.codes = []

    @staticmethod
    def schema(name):
        return importlib.import_module("tflite." + name)

    def tensor(self, name, shape, dtype="float32", data=None):
        buf = 0
        if data is not None:
            self.buffers.append(np.ascontiguousarray(data, dtype=dtype).tobytes())
            buf = len(self.buffers) - 1
        self.tensors.append((name, list(shape), dtype, buf))
        return len(self.tensors) - 1

    def op(self, code, inputs, outputs, options_type=0, options=None):
        if code not in self.codes:
            self.codes.append(code)
        self.ops.append((self.codes.index(code), inputs, outputs, options_type, options))

    def _int_vector(self, b, values):
        b.StartVector(4, len(values), 4)
        for v in reversed(values):
            b.PrependInt32(v)
        return b.EndVector(len(values))

    def serialize(self, inputs, outputs, description):
        s = self.schema
        tensor_type = {"float32": s("TensorType").TensorType.FLOAT32,
                       "int32": s("TensorType").TensorType.INT32}
        b = self.fb.Builder(1 << 20)

        buffer_offsets = []
        for data in self.buffers:
            vec = None
            if data:
                # 16-byte aligned so the runtime can map weights in place.
                b.StartVector(1, len(data), 16)
                b.head -= len(data)
                b.Bytes[b.head:b.head + len(data)] = data
                vec = b.EndVector(len(data))
            m = s("Buffer")
            m.Start(b)
            if vec is not None:
                m.AddData(b, vec)
            buffer_offsets.append(m.End(b))

        tensor_offsets = []
        for name, shape, dtype, buf in self.tensors:
            name_off = b.CreateString(name)
            shape_off = self._int_vector(b, shape)
            m = s("Tensor")
            m.Start(b)
            m.AddShape(b, shape_off)
            m.AddType(b, tensor_type[dtype])
            m.AddBuffer(b, buf)
            m.AddName(b, name_off)
            tensor_offsets.append(m.End(b))

        op_offsets = []
        for code_index, ins, outs, options_type, options in self.ops:
            options_off = options(b) if options else None
            ins_off, outs_off = self._int_vector(b, ins), self._int_vector(b, outs)
            m = s("Operator")
            m.Start(b)
            m.AddOpcodeIndex(b, code_index)
            m.AddInputs(b, ins_off)
            m.AddOutputs(b, outs_off)
            if options_off is not None:
                m.AddBuiltinOptionsType(b, options_type)
                m.AddBuiltinOptions(b, options_off)
            op_offsets.append(m.End(b))

        code_offsets = []
        for code in self.codes:
            m = s("OperatorCode")
            m.Start(b)
            m.AddDeprecatedBuiltinCode(b, min(code, 127))
            m.AddBuiltinCode(b, code)
            m.AddVersion(b, 1)
            code_offsets.append(m.End(b))

        def table_vector(offsets):
            b.StartVector(4, len(offsets), 4)
            for o in reversed(offsets):
                b.PrependUOffsetTRelative(o)
            return b.EndVector(len(offsets))

        tensors_vec, ops_vec = table_vector(tensor_offsets), table_vector(op_offsets)
        in_vec, out_vec = self._int_vector(b, inputs), self._int_vector(b, outputs)
        sg_name = b.CreateString("main")
        m = s("SubGraph")
        m.Start(b)
        m.AddTensors(b, tensors_vec)
        m.AddInputs(b, in_vec)
        m.AddOutputs(b, out_vec)
        m.AddOperators(b, ops_vec)
        m.AddName(b, sg_name)
        subgraph = m.End(b)

        codes_vec = table_vector(code_offsets)
        subgraphs_vec = table_vector([subgraph])
        buffers_vec = table_vector(buffer_offsets)
        desc = b.CreateString(description)
        m = s("Model")
        m.Start(b)
        m.AddVersion(b, 3)
        m.AddOperatorCodes(b, codes_vec)
        m.AddSubgraphs(b, subgraphs_vec)
        m.AddDescription(b, desc)
        m.AddBuffers(b, buffers_vec)
        b.Finish(m.End(b), file_identifier=b"TFL3")
        return bytes(b.Output())


def build_tflite(model, frames, channel_dim, odim):
    w = TfliteWriter()
    s = w.schema
    ops = s("BuiltinOperator").BuiltinOperator
    opt = s("BuiltinOptions").BuiltinOptions
    act = s("ActivationFunctionType").ActivationFunctionType
    valid = s("Padding").Padding.VALID
    count = [0]

    def name(prefix):
        count[0] += 1
        return f"{prefix}_{count[0]}"

    def reshape(x, shape, out_name=None):
        shape_t = w.tensor(name("shape"), [len(shape)], "int32", np.array(shape, np.int32))
        y = w.tensor(out_name or name("reshape"), shape)

        def options(b):
            m = s("ReshapeOptions")
            vec = w._int_vector(b, shape)
            m.Start(b)
            m.AddNewShape(b, vec)
            return m.End(b)

        w.op(ops.RESHAPE, [x, shape_t], [y], opt.ReshapeOptions, options)
        return y

    def linear(x, layer, relu):
        weight = layer.weight.detach().numpy()  # [out, in]
        out_dim = weight.shape[0]
        bias = (layer.bias.detach().numpy() if layer.bias is not None
                else np.zeros(out_dim, np.float32))
        f = w.tensor(name("conv_w"), [out_dim, 1, 1, weight.shape[1]], data=weight[:, None, None, :])
        bt = w.tensor(name("conv_b"), [out_dim], data=bias)
        y = w.tensor(name("conv"), [1, 1, frames, out_dim])

        def options(b):
            m = s("Conv2DOptions")
            m.Start(b)
            m.AddPadding(b, valid)
            m.AddStrideW(b, 1)
            m.AddStrideH(b, 1)
            m.AddDilationWFactor(b, 1)
            m.AddDilationHFactor(b, 1)
            m.AddFusedActivationFunction(b, act.RELU if relu else act.NONE)
            return m.End(b)

        w.op(ops.CONV_2D, [x, f, bt], [y], opt.Conv2DOptions, options)
        return y

    def fsmn(x, layer):
        kernel, left, right = fsmn_kernel(layer)
        channels = kernel.shape[1]
        pads = w.tensor(name("pads"), [4, 2], "int32",
                        np.array([[0, 0], [0, 0], [left, right], [0, 0]], np.int32))
        padded = w.tensor(name("pad"), [1, 1, frames + left + right, channels])

        def pad_options(b):
            m = s("PadOptions")
            m.Start(b)
            return m.End(b)

        w.op(ops.PAD, [x, pads], [padded], opt.PadOptions, pad_options)
        f = w.tensor(name("fsmn_w"), [1, 1, kernel.shape[0], channels], data=kernel[None, None])
        bt = w.tensor(name("fsmn_b"), [channels], data=np.zeros(channels, np.float32))
        y = w.tensor(name("fsmn"), [1, 1, frames, channels])

        def options(b):
            m = s("DepthwiseConv2DOptions")
            m.Start(b)
            m.AddPadding(b, valid)
            m.AddStrideW(b, 1)
            m.AddStrideH(b, 1)
            m.AddDepthMultiplier(b, 1)
            m.AddDilationWFactor(b, 1)
            m.AddDilationHFactor(b, 1)
            m.AddFusedActivationFunction(b, act.NONE)
            return m.End(b)

        w.op(ops.DEPTHWISE_CONV_2D, [padded, f, bt], [y], opt.DepthwiseConv2DOptions, options)
        return y

    def add(x, y):
        z = w.tensor(name("add"), w.tensors[x][1])

        def options(b):
            m = s("AddOptions")
            m.Start(b)
            m.AddFusedActivationFunction(b, act.NONE)
            return m.End(b)

        w.op(ops.ADD, [x, y], [z], opt.AddOptions, options)
        return z

    d = model.dfsmn
    feat = w.tensor("feat", [1, frames, 80, 1] if channel_dim else [1, frames, 80])
    x = reshape(feat, [1, 1, frames, 80])
    x = linear(x, d.fc1[0], relu=True)
    x = linear(x, d.fc2[0], relu=True)
    m = fsmn(x, d.fsmn1)
    for block in d.fsmns:
        h = linear(m, block.fc1[0], relu=True)
        p = linear(h, block.fc2, relu=False)
        m = add(fsmn(p, block.fsmn), m)
    for layer in d.dnns:
        if isinstance(layer, torch.nn.Linear):
            m = linear(m, layer, relu=True)  # every DNN Linear is followed by ReLU
    logits = linear(m, model.out, relu=False)
    sig = w.tensor(name("sigmoid"), [1, 1, frames, odim])
    w.op(ops.LOGISTIC, [logits], [sig])
    probs = reshape(sig, [1, frames, odim], out_name="probs")
    return w.serialize([feat], [probs], f"FireRedVAD fixed T={frames}")


def check_tflite(path, module, shape):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as e:  # also covers DLLs blocked by Windows Application Control
        print(f"{path}: not checked, ai-edge-litert unavailable ({e})")
        return
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    feat = speech_like_feat(shape)
    interp.set_tensor(inp["index"], feat)
    interp.invoke()
    got = interp.get_tensor(out["index"]).reshape(-1)
    want = module(torch.from_numpy(feat)).numpy().reshape(-1)
    print(f"{path}: LiteRT vs PyTorch max|diff| {np.abs(got - want).max():.2e}, "
          f"decisions flipped at 0.5: {int(((got > 0.5) != (want > 0.5)).sum())}/{got.size}")


def speech_like_feat(shape, seed=0):
    # A random walk switched on and off, so probabilities span 0..1.
    rng = np.random.default_rng(seed)
    t = shape[1]
    walk = np.cumsum(rng.normal(size=(t, 80)), 0)
    walk = (walk - walk.mean(0)) / (walk.std(0) + 1e-6)
    gate = (np.sin(np.arange(t) / 170) > 0)[:, None]
    return (0.6 * walk + gate * rng.normal(1, 0.5, size=(t, 80))).astype(np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="FireRedVAD source checkout (default: installed package)")
    ap.add_argument("--ckpt", required=True, help="VAD/model.pth.tar or AED/model.pth.tar")
    ap.add_argument("--frames", type=int, required=True, help="fixed T (10 ms frames)")
    ap.add_argument("--channel-dim", action="store_true", help="input [1, T, 80, 1] instead of [1, T, 80]")
    ap.add_argument("--out", required=True, nargs="+", help="output files, .onnx and/or .tflite")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    for out in args.out:
        if os.path.splitext(out)[1] not in (".onnx", ".tflite"):
            raise SystemExit(f"{out}: extension must be .onnx or .tflite")

    torch.serialization.add_safe_globals([argparse.Namespace])
    package = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    a = package["args"]
    model = load_detect_model_class(args.repo)(a)
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()

    # Each of the R FSMN layers sees (N1-1)*S1 past and N2*S2 future frames.
    lookback, lookahead = a.R * (a.N1 - 1) * a.S1, a.R * a.N2 * a.S2
    print(f"DFSMN R={a.R} N1={a.N1} S1={a.S1} N2={a.N2} S2={a.S2} odim={a.odim}: "
          f"receptive field {lookback} past + {lookahead} future frames")
    if args.frames <= lookback + lookahead:
        print(f"warning: T={args.frames} <= {lookback + lookahead}, "
              "windowed results will differ from the dynamic model")

    shape = (1, args.frames, 80, 1) if args.channel_dim else (1, args.frames, 80)
    module = NonStreaming(model, args.frames, args.channel_dim)
    for out in args.out:
        if out.endswith(".onnx"):
            torch.onnx.export(
                module, (torch.randn(*shape),), out,
                input_names=["feat"], output_names=["probs"],
                opset_version=args.opset, dynamo=False,
            )
        else:
            with open(out, "wb") as f:
                f.write(build_tflite(model, args.frames, args.channel_dim, a.odim))
        print(f"{out}: feat {list(shape)} -> probs [1, {args.frames}, {a.odim}], "
              f"{os.path.getsize(out):,} bytes")
        if out.endswith(".tflite"):
            check_tflite(out, module, shape)


if __name__ == "__main__":
    main()
