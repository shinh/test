import onnx
from onnx import onnx_pb
import sys


def find_subgraphs(graph):
    subgraphs = {}
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx_pb.AttributeProto.GRAPH:
                for name, graph in find_subgraphs(attr.g).items():
                    subgraphs[name] = graph
                subgraphs[attr.g.name] = attr.g
    return subgraphs


def main():
    xmodel = onnx.load(sys.argv[1])
    subgraphs = find_subgraphs(xmodel.graph)
    if len(sys.argv) == 2:
        for name, graph in subgraphs.items():
            print('name=%s node=%d' % (name, len(graph.node)))
    else:
        g = subgraphs[sys.argv[2]]
        m = onnx.helper.make_model(g)
        onnx.save(m, 'out.onnx')


main()
