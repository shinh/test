import onnx

model = onnx.load("albert.onnx")
inits = {}
bases = {}
for init in model.graph.initializer:
    inits[init.name] = init
    bases[init.name] = init.name

for node in model.graph.node:
    if node.op_type == "Identity":
        init = inits.get(node.input[0])
        if init is not None:
            inits[node.output[0]] = init
            bases[node.output[0]] = bases[node.input[0]]

    if node.op_type == "MatMul":
        init = inits.get(node.input[1])
        if init is not None:
            print(bases[init.name], init.dims)