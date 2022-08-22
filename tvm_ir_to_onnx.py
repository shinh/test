import sys

import numpy as np
import onnx
import tvm


def save_ir_as_onnx(mod, out):
    assert len(mod.functions) == 1
    func = list(mod.functions.values())[0]
    body = func.body

    queue = [body]
    node_to_idx = {}
    nodes = []

    def visit(node):
        node_id = str(node.handle)
        if node_id in node_to_idx:
            return

        if hasattr(node, "args"):
            for arg in node.args:
                visit(arg)

        nodes.append(node)
        node_to_idx[node_id] = len(nodes)

    visit(body)

    node_to_name = {}
    onnx_nodes = []
    for idx, node in enumerate(nodes):
        if isinstance(node, tvm.relay.Constant):
            name = f"c{idx}"
            node_to_name[str(node.handle)] = name
            tensor = onnx.numpy_helper.from_array(node.data.numpy(), name=name)
            onnx_nodes.append(onnx.helper.make_node(
                "Constant", [], [name], value=tensor
            ))
            continue

        if isinstance(node, tvm.relay.Var):
            name = str(node.name_hint)
            node_to_name[str(node.handle)] = name
            continue

        assert isinstance(node, tvm.relay.Call), str(node)

        name = f"t{idx}"
        node_to_name[str(node.handle)] = name

        onnx_inputs = []
        for arg in node.args:
            assert str(arg.handle) in node_to_name, str(arg)
            onnx_inputs.append(node_to_name[str(arg.handle)])

        def to_prim(value):
            if isinstance(value, (int, float, str)):
                return value
            if isinstance(value, tvm.runtime.container.String):
                return str(value)
            if isinstance(value, tvm.tir.expr.IntImm):
                return int(value)
            if isinstance(value, (list, tvm.ir.container.Array)):
                return [to_prim(v) for v in value]
            assert False, f"Unknown type: {type(value)}"

        onnx_attrs = {}
        for key in [] if node.attrs is None else node.attrs.keys():
            value = to_prim(node.attrs[key])
            onnx_attrs[key] = value

        onnx_nodes.append(onnx.helper.make_node(
            str(node.op), onnx_inputs, [name], **onnx_attrs
        ))

    value_infos = []
    for node in nodes:
        name = node_to_name[str(node.handle)]

        try:
            typ = node.checked_type
        except ValueError:
            continue

        onnx_dtype = onnx.TensorProto.UNDEFINED
        if hasattr(np, typ.dtype):
            np_dtype = np.dtype(getattr(np, typ.dtype))
            if np_dtype in onnx.mapping.NP_TYPE_TO_TENSOR_TYPE:
                onnx_dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_dtype]
        onnx_shape = [int(dim) for dim in typ.shape]

        onnx_type = onnx.helper.make_tensor_type_proto(onnx_dtype, onnx_shape)
        value_infos.append(onnx.helper.make_value_info(name, onnx_type))

    onnx_output = onnx.ValueInfoProto(name=node_to_name[str(body.handle)])
    onnx_graph = onnx.helper.make_graph(onnx_nodes, out, [], [onnx_output], value_info=value_infos)

    onnx_model = onnx.helper.make_model(onnx_graph)
    onnx.save(onnx_model, out)


def main():
    with open(sys.argv[1]) as f:
        mod = tvm.ir.load_json(f.read())
    save_ir_as_onnx(mod, sys.argv[2])


if __name__ == "__main__":
    main()
