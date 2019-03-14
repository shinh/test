#!/usr/bin/python3

import glob
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper


dest_np_type = np.int8
dest_onnx_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dest_np_type)]


def convert_type(type):
    assert type.tensor_type
    type.tensor_type.elem_type = dest_onnx_type


def convert_tensor(tensor):
    a = numpy_helper.to_array(tensor)
    a = a / max(abs(np.max(a)), abs(np.min(a))) * 127
    a = a.astype(dest_np_type)
    tensor.CopyFrom(numpy_helper.from_array(a, name=tensor.name))


def convert_value_info(value):
    convert_type(value.type)


def convert_model(xmodel):
    for initializer in xmodel.graph.initializer:
        convert_tensor(initializer)
    for value in xmodel.graph.input:
        convert_value_info(value)
    for value in xmodel.graph.output:
        convert_value_info(value)
    for value in xmodel.graph.value_info:
        convert_value_info(value)


def main():
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]

    os.makedirs(to_dir, exist_ok=True)

    xmodel = onnx.load(os.path.join(from_dir, 'model.onnx'))
    convert_model(xmodel)
    onnx.save(xmodel, os.path.join(to_dir, 'model.onnx'))

    for test_dir in glob.glob(os.path.join(from_dir, 'test_data_set_*')):
        dir_name = os.path.basename(test_dir)
        to_test_dir = os.path.join(to_dir, dir_name)
        os.makedirs(to_test_dir, exist_ok=True)

        for pb_filename in glob.glob(os.path.join(test_dir, '*.pb')):
            pb_name = os.path.basename(pb_filename)
            tensor = onnx.load_tensor(pb_filename)
            convert_tensor(tensor)
            onnx.save_tensor(tensor, os.path.join(to_test_dir, pb_name))


main()
