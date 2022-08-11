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
            name = f"i{idx}"
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

    onnx_output = onnx.ValueInfoProto(name=node_to_name[str(body.handle)])
    onnx_graph = onnx.helper.make_graph(onnx_nodes, out, [], [onnx_output])

    onnx_model = onnx.helper.make_model(onnx_graph)
    onnx.save(onnx_model, out)
