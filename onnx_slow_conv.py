import sys

import onnx
from onnx import numpy_helper

num_split = 2

if len(sys.argv) != 3:
    raise RuntimeError('Usage: %s input.onnx output.onnx' % sys.argv[0])


model = onnx.load(sys.argv[1], 'rb')

params = {}
for init in model.graph.initializer:
    params[init.name] = numpy_helper.to_array(init)

new_nodes = []
split_params = set()

for node in model.graph.node:
    if node.op_type != 'Conv':
        new_nodes.append(node)
        continue

    assert len(node.output) == 1
    output = node.output[0]
    for input in node.input[1:]:
        split_params.add(input)

    tmp_names = []
    for i in range(num_split):
        tmp_name = '%s_sub_%d' % (output, i)
        tmp_names.append(tmp_name)
        n = onnx.helper.make_node('Conv',
                                  inputs=node.input,
                                  outputs=[tmp_name])
        for a in node.attribute:
            n.attribute.add().CopyFrom(a)
        new_nodes.append(n)
    n = onnx.helper.make_node('Sum',
                              inputs=tmp_names,
                              outputs=[output])
    new_nodes.append(n)

for name in split_params:
    params[name] = params[name] / num_split

while model.graph.node:
    model.graph.node.pop()

for node in new_nodes:
    model.graph.node.add().CopyFrom(node)

while model.graph.initializer:
    model.graph.initializer.pop()

for name, param in params.items():
    init = model.graph.initializer.add()
    init.CopyFrom(numpy_helper.from_array(param, name=name))

onnx.save(model, sys.argv[2])
