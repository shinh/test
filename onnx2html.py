import argparse
import collections
import html
import sys

import onnx
import onnx.numpy_helper
import onnx.shape_inference


def cleanse(name):
    return name


def anchor(text, label):
    return f'<a href="#{label}">{text}</a>'


def graph_detail_label(graph_name):
    return f"gd_{graph_name}"


def node_detail_label(node):
    return f"nd_{node.output[0]}"


def node_summary_label(node):
    return f"ns_{node.output[0]}"


def value_detail_label(value):
    return f"vd_{value}"


Value = collections.namedtuple(
    "Value",
    ("name", "producer", "users", "kind", "info")
)


def create_value_map(graph):
    value_to_info = {}
    for vi in graph.value_info:
        value_to_info[vi.name] = ("temp", vi)
    for vi in graph.input:
        value_to_info[vi.name] = ("input", vi)
    for tensor in graph.initializer:
        vi = onnx.helper.make_tensor_value_info(
            tensor.name, tensor.data_type, tensor.dims
        )
        value_to_info[vi.name] = ("init", vi)
    for vi in graph.output:
        value_to_info[vi.name] = ("output", vi)

    for node in graph.node:
        for value in list(node.input) + list(node.output):
            if value not in value_to_info:
                value_to_info[value] = ("temp", onnx.ValueInfoProto(name=value))

    value_to_producer = {}
    value_to_users = {}
    for node in graph.node:
        for input in node.input:
            if input not in value_to_users:
                value_to_users[input] = []
            value_to_users[input].append(node)
        for output in node.output:
            assert output not in value_to_producer, "Not SSA"
            value_to_producer[output] = node

    value_map = {}
    for name, (kind, info) in value_to_info.items():
        producer = value_to_producer.get(name)
        users = value_to_users.get(name, [])
        value_map[name] = Value(name, producer, users, kind, info)
    return value_map


def dtype_to_str(dtype):
    return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(dtype, "UnknownDtype")


def shape_to_str(shape):
    strs = []
    for dim in shape.dim:
        if dim.dim_value is not None:
            strs.append(str(dim.dim_value))
        elif dim.dim_param is not None:
            strs.append(dim.dim_param)
        else:
            strs.append("???")
    return "(" + ",".join(strs) + ")"


def type_to_str(typ):
    if typ.HasField("tensor_type"):
        dtype = dtype_to_str(typ.tensor_type.elem_type)
        shape = shape_to_str(typ.tensor_type.shape)
        return f"{dtype} {shape}"

    if typ.HasField("sequence_type"):
        return f"Sequence({type_to_str(typ.sequence_type.elem_type)})"

    if typ.HasField("map_type"):
        key_str = dtype_to_str(typ.map_type.key_type)
        value_str = type_to_str(typ.map_type.value_type)
        return f"Map({key_str},{value_str})"

    if typ.HasField("optional_type"):
        return f"Optional({type_to_str(typ.sequence_type.elem_type)})"

    if typ.HasField("sparse_tensor_type"):
        dtype = dtype_to_str(typ.tensor_type.elem_type)
        shape = shape_to_str(typ.tensor_type.shape)
        return f"{dtype} {shape}"

    return "UnknownType"


def node_summary_str(node):
    op_str = anchor(node.op_type, node_detail_label(node))
    inputs_str = ", ".join(anchor(v, value_detail_label(v)) for v in node.input)
    outputs_str = ", ".join(anchor(v, value_detail_label(v)) for v in node.output)
    return "{}({}) -> ({})".format(op_str, inputs_str, outputs_str)


def value_summary_str(vi):
    type_str = type_to_str(vi.type)
    name_str = anchor(vi.name, value_detail_label(vi.name))
    return f"{name_str} {type_str}"


def attr_value_str(value):
    if isinstance(value, list):
        return [attr_value_str(v) for v in value]
    if isinstance(value, onnx.GraphProto):
        return anchor("subgraph", graph_detail_label(value.name))
    if isinstance(value, onnx.TensorProto):
        numel = 1
        for dim in value.dims:
            numel *= dim
        dtype = dtype_to_str(value.data_type)
        shape = list(value.dims)
        if numel > 50:
            return f"Tensor(dtype={dtype} shape={shape})"
        array = onnx.numpy_helper.to_array(value)
        return f"Tensor(dtype={dtype} shape={shape} value={array})"
    if isinstance(value, bytes):
        return value.decode()
    return value


def graph_to_str(graph, graph_name, parent_name=None):
    value_map = create_value_map(graph)

    html_str = ""

    html_str += f'<h1 id="{graph_detail_label(graph_name)}">{graph_name}</h1>'

    if parent_name:
        html_str += "Parent graph: " + anchor(parent_name, graph_detail_label(parent_name))

    html_str += f'<h2>Graph inputs</h2>'
    html_str += "<ul>"
    for input in graph.input:
        html_str += "<li>" + value_summary_str(input)
    html_str += "</ul>"

    html_str += f'<h2>Graph outputs</h2>'
    html_str += "<ul>"
    for output in graph.output:
        html_str += "<li>" + value_summary_str(output)
    html_str += "</ul>"

    html_str += "<h2>=== node summary ===</h2><ul>"
    for node in graph.node:
        node_str = node_summary_str(node)
        html_str += f'<li id="{node_summary_label(node)}">' + node_str
    html_str += "</ul>"

    html_str += "<h2>=== value detail ===</h2>"
    for value in value_map.values():
        value_str = value_summary_str(value.info)
        html_str += f'<h3 id="{value_detail_label(value.name)}">{value_str} {value.kind}</h3>'
        html_str += "<ul>"
        if value.producer:
            html_str += f"<li>producer: " + node_summary_str(value.producer)
        for user in value.users:
            html_str += f"<li>user: " + node_summary_str(user)
        html_str += "</ul>"

    subgraphs = []

    for node in graph.node:
        node_name = node.op_type
        if node.name:
            node_name += f" ({node.name})"
        node_str = anchor(node_name, node_summary_label(node))
        html_str += f'<h3 id="{node_detail_label(node)}">{node_str}</h3>'

        if node.input:
            html_str += f'<h3>inputs</h2>'
        html_str += "<ul>"
        for input in node.input:
            if input == "":
                html_str += "<li>(null)"
            else:
                html_str += "<li>" + value_summary_str(value_map[input].info)
        html_str += "</ul>"

        html_str += f'<h3>outputs</h2>'
        html_str += "<ul>"
        for output in node.output:
            html_str += "<li>" + value_summary_str(value_map[output].info)
        html_str += "</ul>"

        if node.attribute:
            html_str += f'<h3>attributes</h2>'
        html_str += "<ul>"
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            value_str = attr_value_str(value)
            if isinstance(value, onnx.GraphProto):
                subgraphs.append(value)
            html_str += f"<li>{attr.name}: {value_str}"
        html_str += "</ul>"

        if node.doc_string:
            html_str += f'<h3>doc_string</h2>'
            html_str += "<pre>{node.doc_string}</pre>"

    for subgraph in subgraphs:
        html_str += graph_to_str(subgraph, subgraph.name, graph_name)

    return html_str


def onnx2html(model):
    model = onnx.shape_inference.infer_shapes(model)

    html_str = graph_to_str(model.graph, "Top level graph")

    return html_str


def main():
    parser = argparse.ArgumentParser(description="ONNX to HTML")
    parser.add_argument("onnx")
    parser.add_argument("html")
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    html = onnx2html(model)
    with open(args.html, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
