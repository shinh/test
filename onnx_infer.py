import sys
import onnx
import onnx.shape_inference

if len(sys.argv) != 3:
    raise RuntimeError('Usage: %s input.onnx output.onnx' % sys.argv[0])

model = onnx.load(sys.argv[1], 'rb')
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, sys.argv[2])
