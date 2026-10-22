# Replace every If whose condition is constant once sr=16000 by the taken branch.
import sys, onnx, numpy as np, onnxruntime as ort
from onnx import helper, numpy_helper
src, dst = sys.argv[1:3]
m = onnx.load(src); g = m.graph
for i in list(g.input):
    if i.name == "sr": g.input.remove(i)
g.initializer.append(numpy_helper.from_array(np.array(16000, np.int64), "sr"))

def cond_value(model, name):
    mm = onnx.ModelProto(); mm.CopyFrom(model)
    mm.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.BOOL, None))
    s = ort.InferenceSession(mm.SerializeToString())
    feeds = {"input": np.zeros((1, 576), np.float32), "state": np.zeros((2, 1, 128), np.float32)}
    feeds = {k: v for k, v in feeds.items() if k in [i.name for i in s.get_inputs()]}
    return bool(s.run([name], feeds)[0])

while True:
    ifs = [n for n in g.node if n.op_type == "If"]
    if not ifs: break
    n = ifs[0]
    branch = {a.name: a.g for a in n.attribute}["then_branch" if cond_value(m, n.input[0]) else "else_branch"]
    idx = list(g.node).index(n)
    new = list(branch.node) + [helper.make_node("Identity", [bo.name], [o]) for bo, o in zip(branch.output, n.output)]
    g.initializer.extend(branch.initializer)
    g.node.remove(n)
    for k, nn in enumerate(new): g.node.insert(idx + k, nn)
for i in g.input:
    d = i.type.tensor_type.shape.dim
    if i.name == "input": d[0].dim_value = 1; d[1].dim_value = 576
    if i.name == "state": d[0].dim_value = 2; d[1].dim_value = 1; d[2].dim_value = 128
onnx.checker.check_model(m)
from onnxsim import simplify
m, ok = simplify(m); print("simplified", ok, sorted({n.op_type for n in m.graph.node}))
onnx.save(m, dst)
