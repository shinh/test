import argparse
import glob
import os
import sys

import nnvm
import nnvm.compiler
import numpy as np
import onnx
import onnx.numpy_helper
import tvm

from tvm.contrib import graph_runtime


def load_test_data(data_dir):
    inout_values = []
    for kind in ['input', 'output']:
        values = []
        for pb in sorted(glob.glob(os.path.join(data_dir, '%s_*.pb' % kind))):
            with open(pb, 'rb') as f:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(f.read())
            values.append((tensor.name, onnx.numpy_helper.to_array(tensor)))
        inout_values.append(values)
    return tuple(inout_values)


def compile(symbol, target, input_names, inputs, params):
    shape_dict = {}
    dtype_dict = {}
    for name, value in zip(input_names, inputs.values()):
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    for name, value in params.items():
        shape_dict[name] = value.shape
        dtype_dict[name] = value.dtype
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(symbol, target,
                                                 shape=shape_dict,
                                                 dtype=dtype_dict,
                                                 params=params)
    return graph, lib, params


def run(args):
    test_data_dir = os.path.join(args.test_dir, 'test_data_set_0')
    inputs, outputs = load_test_data(test_data_dir)
    inputs = dict(inputs)

    onnx_model = onnx.load_model(os.path.join(args.test_dir, 'model.onnx'))
    symbol, params = nnvm.frontend.from_onnx(onnx_model)
    input_names = symbol.list_input_names()
    output_names = symbol.list_output_names()

    assert len(input_names) == len(inputs) + len(params)
    # assert len(output_names) == len(outputs)

    graph, lib, params = compile(
        symbol, args.target, input_names, inputs, params)

    if args.dump_nnvm:
        print(graph.json())

    ctx = tvm.gpu()

    # Prepare inputs.
    tvm_inputs = {k: tvm.nd.array(v, ctx=ctx) for k, v in inputs.items()}
    tvm_params = {k: tvm.nd.array(v, ctx=ctx) for k, v in params.items()}

    graph_module = graph_runtime.create(graph, lib, ctx)

    graph_module.set_input(**tvm_inputs)
    graph_module.set_input(**tvm_params)

    graph_module.run()

    for i, (name, expected) in enumerate(outputs):
        tvm_output = tvm.nd.empty(expected.shape, expected.dtype, ctx=ctx)
        actual = graph_module.get_output(i, tvm_output).asnumpy()
        np.testing.assert_allclose(expected, actual,
                                   rtol=1e-3, atol=1e-4), name
        print('%s: OK' % name)
    print('ALL OK')


def main():
    parser = argparse.ArgumentParser(description='Run ONNX by TVM')
    parser.add_argument('test_dir')
    parser.add_argument('--dump_nnvm', action='store_true')
    parser.add_argument('--target', type=str, default='cuda')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
