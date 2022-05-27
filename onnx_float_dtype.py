import argparse
import cmath
import struct
import sys

import onnx
import onnx.mapping
import onnx.numpy_helper


# From https://github.com/onnx/onnx/pull/4193/files
# convert a f32 to bf16 (as int)
def float32_to_bfloat16(fval: float) -> int:
    ival = int.from_bytes(struct.pack('<f', fval), 'little')
    if cmath.isnan(fval):
        # NaN requires at least 1 significand bit set
        ival16 = 0x7FC0  # sign=0, exp=all-ones, sig=0b1000000
    else:
        # drop bottom 16-bits
        # round remaining bits using round-to-nearest-even
        round = ((ival >> 16) & 1) + 0x7fff
        ival16 = (ival + round) >> 16
    # swap byte order for big-endian
    if sys.byteorder == 'big':
        bytes = struct.pack('<h', ival16)
        ival16 = int.from_bytes(bytes, 'big')
    return ival16


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
        tensor.int32_data.extend([float32_to_bfloat16(v) for v in a.flatten()])
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
