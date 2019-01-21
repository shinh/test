import sys
import onnx
import onnx.optimizer
if len(sys.argv) != 3:
    raise RuntimeError('Usage: %s input.onnx output.onnx' % sys.argv[0])
passes = [
    'eliminate_identity',
    # 'eliminate_nop_transpose',
    # 'eliminate_nop_pad',
    # 'eliminate_unused_initializer',
    'fuse_bn_into_conv',
    # 'fuse_consecutive_squeezes',
    # 'fuse_consecutive_transposes',
    # 'fuse_add_bias_into_conv',
    # 'fuse_transpose_into_gemm',
]
model = onnx.ModelProto()
model.ParseFromString(open(sys.argv[1], 'rb').read())
model = onnx.optimizer.optimize(model, passes=passes)
with open(sys.argv[2], 'wb') as f:
    f.write(model.SerializeToString())
