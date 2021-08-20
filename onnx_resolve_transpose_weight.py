import sys

import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1])

params = {}
for idx, param in enumerate(m.graph.initializer):
    params[param.name] = idx

for node in m.graph.node:
    if node.op_type != "Gemm":
        continue

    trans_b_idx = None
    for idx, attr in enumerate(node.attribute):
        if attr.name == "transB":
            trans_b_idx = idx
    if trans_b_idx is None:
        continue

    del node.attribute[trans_b_idx]
    param_idx = params[node.input[1]]
    weight = m.graph.initializer[param_idx]
    wt = numpy_helper.to_array(weight).T
    print("Resolve transB in %s for %s" % (node.name, weight.name))
    print(weight.dims, "=>", wt.shape)
    new_weight = numpy_helper.from_array(wt, weight.name)
    m.graph.initializer[param_idx].dims[:] = new_weight.dims
    m.graph.initializer[param_idx].raw_data = new_weight.raw_data

onnx.save(m, sys.argv[2])
