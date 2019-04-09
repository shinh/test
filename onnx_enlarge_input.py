import os
import random
import shutil
import sys

import numpy as np
import onnx
import onnx.numpy_helper


if len(sys.argv) != 4:
    raise RuntimeError('Usage: %s input_dir output_dir ratio' % sys.argv[0])


input_dir = sys.argv[1]
output_dir = sys.argv[2]
ratio = float(sys.argv[3])

os.makedirs(output_dir, exist_ok=True)
shutil.rmtree(os.path.join(output_dir, 'test_data_set_0'), ignore_errors=True)
shutil.copytree(os.path.join(input_dir, 'test_data_set_0'),
                os.path.join(output_dir, 'test_data_set_0'))

model = onnx.load(os.path.join(input_dir, 'model.onnx'))
initializer_names = {i.name for i in model.graph.initializer}

inputs = []
for input in model.graph.input:
    if input.name not in initializer_names:
        inputs.append(input)

assert len(inputs) == 1

input = inputs[0]
old_shape = []
new_shape = []
for i in range(len(input.type.tensor_type.shape.dim)):
    nd = input.type.tensor_type.shape.dim[i].dim_value
    old_shape.append(nd)
    if i >= 2:
        nd = int(nd * ratio)
    new_shape.append(nd)
    input.type.tensor_type.shape.dim[i].dim_value = nd

print('new_shape', new_shape)

onnx.save(model, os.path.join(output_dir, 'model.onnx'))

input_pb = os.path.join(output_dir, 'test_data_set_0/input_0.pb')
input_tensor = onnx.load_tensor(input_pb)
assert len(input_tensor.dims) == len(new_shape)

input = onnx.numpy_helper.to_array(input_tensor)

pad_width = []
for od, nd in zip(old_shape, new_shape):
    pad_width.append((0, nd - od))

input = np.pad(input, pad_width, 'constant')

onnx.save_tensor(onnx.numpy_helper.from_array(input), input_pb)
