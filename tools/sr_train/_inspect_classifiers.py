import sys
import onnx
from collections import Counter

paths = {
    "noise_det-v1": r"C:\TopazModels\noise_det-v1-fp16-128x128-1x-ox\noise_det-v1-fp16.onnx",
    "liting_det-v1": r"C:\TopazModels\liting_det-v1-fp16-128x128-ox\liting_det-v1-fp16.onnx",
    "gendet-v1-ox": r"C:\TopazModels\gendet-v1-fp32-256x256-1x-ox\gendet-v1-fp32-256x256-1x.onnx",
}

for name, p in paths.items():
    print(f"\n===== {name} ({p}) =====")
    m = onnx.load(p)
    g = m.graph
    print("ir_version", m.ir_version, "opset", [(o.domain, o.version) for o in m.opset_import])
    print("inputs:")
    for inp in g.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(" ", inp.name, inp.type.tensor_type.elem_type, dims)
    print("outputs:")
    for out in g.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(" ", out.name, out.type.tensor_type.elem_type, dims)
    ops = Counter(n.op_type for n in g.node)
    print("op_type counts:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print("num_nodes:", len(g.node))
    # print first few and last few node op_types in order to see overall shape
    seq = [n.op_type for n in g.node]
    print("node sequence head:", seq[:15])
    print("node sequence tail:", seq[-15:])
