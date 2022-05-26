import argparse

import onnx
import onnx.mapping
import onnx.numpy_helper


def _modify_type_proto(typ, dtype):
    if not typ.tensor_type:
        raise RuntimeError("Only Tensor type is supported")

    if typ.tensor_type.elem_type == onnx.TensorProto.FLOAT:
        typ.tensor_type.elem_type = dtype


def _modify_tensor_proto(tensor, dtype):
    if tensor.data_type != onnx.TensorProto.FLOAT:
        return

    a = onnx.numpy_helper.to_array(tensor)

    if dtype == onnx.TensorProto.BFLOAT16:
        # No numpy type for bfloat16.
        tensor.ClearField("raw_data")
        tensor.ClearField("float_data")
        tensor.float_data.extend(list(a.flatten()))
        tensor.data_type = dtype
        return

    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]
    a = a.astype(np_dtype)
    tensor.CopyFrom(onnx.numpy_helper.from_array(a, tensor.name))


def main():
    parser = argparse.ArgumentParser(description="Run tests on decomposed ONNX layers")
    parser.add_argument("in_onnx", help="The input ONNX model")
    parser.add_argument("out_onnx", help="The output ONNX model")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    model = onnx.load(args.in_onnx)

    assert hasattr(onnx.TensorProto, args.dtype.upper()), "dtype=%s not supported" % args.dtype
    dtype = getattr(onnx.TensorProto, args.dtype.upper())

    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        _modify_type_proto(value.type, dtype)

    for init in model.graph.initializer:
        _modify_tensor_proto(init, dtype)

    onnx.save(model, args.out_onnx)


main()
