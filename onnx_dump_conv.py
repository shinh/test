import sys
import onnx
import onnx.shape_inference

if len(sys.argv) != 2:
    raise RuntimeError('Usage: %s input.onnx' % sys.argv[0])

model = onnx.load(sys.argv[1], 'rb')
model = onnx.shape_inference.infer_shapes(model)

value_infos = {}
for vi in model.graph.input:
    value_infos[vi.name] = vi
for vi in model.graph.output:
    value_infos[vi.name] = vi
for vi in model.graph.value_info:
    value_infos[vi.name] = vi

total_kflops = 0

for node in model.graph.node:
    if node.op_type != 'Conv':
        continue
    if (node.input[0] in value_infos and
        node.input[1] in value_infos and
        node.output[0] in value_infos):
        in_shape = value_infos[node.input[0]].type.tensor_type.shape
        w_shape = value_infos[node.input[1]].type.tensor_type.shape
        out_shape = value_infos[node.output[0]].type.tensor_type.shape

        kvs = [
            ('bs', in_shape.dim[0].dim_value),
            ('kw', w_shape.dim[2].dim_value),
            ('kh', w_shape.dim[3].dim_value),
            ('ic', in_shape.dim[1].dim_value),
            ('iw', in_shape.dim[2].dim_value),
            ('ih', in_shape.dim[3].dim_value),
            ('oc', out_shape.dim[1].dim_value),
            ('ow', out_shape.dim[2].dim_value),
            ('oh', out_shape.dim[3].dim_value),
        ]

        groups = 1
        for attr in node.attribute:
            if attr.name == 'group':
                groups = attr.i
        kvs.append(('g', groups))
        m = dict(kvs)
        kflops = (m['bs'] * m['ic'] * m['oc'] * m['ow'] * m['oh'] *
                  m['kw'] * m['kh'] // m['g'] // 1000);
        total_kflops += kflops
        kvs.append(('kflops', kflops))

        print(' '.join(['%s=%s' % (k, v) for k, v in kvs]))
    else:
        print('Unknown Conv(%s, %s) -> (%s)' %
              (tuple(node.input) + tuple(node.output)))

print('Total MFLOPs:', total_kflops // 1000)
