# Evaluate ONNX node by onnxruntime.
#
# Usage:
#
# import numpy as np
# import onnx_eval
# ec = onnx_eval.EvalContext()
# print(ec.Mul(np.arange(3).reshape((1, 3)), np.arange(3).reshape((3, 1))))
# # [[0 0 0]
# #  [0 1 2]
# #  [0 2 4]]


import onnx
import onnxruntime


def _make_value_info(name):
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def _extract_value_info(arr, name):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape)


class EvalContext(object):
    def __init__(self, opset_version=None):
        self.opset_version = opset_version

    def __getattr__(self, op):
        def fn(*args, outs=1, **kwargs):
            input_names = ['input_%d' % i for i in range(len(args))]
            output_names = ['output_%d' % i for i in range(outs)]
            input_vis = [_extract_value_info(a, n) for n, a in
                         zip(input_names, args)]
            output_vis = [_make_value_info(n) for n in output_names]
            node = onnx.helper.make_node(op,
                                         inputs=input_names,
                                         outputs=output_names,
                                         **kwargs)
            graph = onnx.helper.make_graph([node],
                                           'onnx_eval',
                                           inputs=input_vis,
                                           outputs=output_vis)

            opset_imports = None
            if self.opset_version is not None:
                opset_imports = [onnx.helper.make_operatorsetid(
                    '', self.opset_version)]
            model = onnx.helper.make_model(graph, opset_imports=opset_imports)
            serialized = model.SerializeToString()

            session = onnxruntime.InferenceSession(serialized)
            inputs = dict(zip(input_names, args))
            outputs = session.run(output_names, inputs)
            if outs == 1:
                return outputs[0]
            else:
                return outputs

        return fn
