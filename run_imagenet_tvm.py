# Based on https://github.com/dmlc/tvm/blob/master/tutorials/nnvm/from_onnx.py

"""
Compile ONNX Models
===================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy ONNX models with NNVM.

For us to begin with, onnx module is required to be installed.

A quick solution is to install protobuf compiler, and

.. code-block:: bash

    pip install onnx --user

or please refer to offical site.
https://github.com/onnx/onnx
"""

import sys
import timeit

import nnvm
import nnvm.compiler
import tvm
import onnx
import numpy as np

from tvm.contrib import graph_runtime

onnx_model = onnx.load_model(sys.argv[1])
# we can load the graph as NNVM compatible model
sym, params = nnvm.frontend.from_onnx(onnx_model)

x = np.ones([1, 3, 224, 224], dtype=np.float32)

######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.

#target = 'cuda'
target = 'llvm'
# assume first input name is data
input_name = sym.list_input_names()[0]
shape_dict = {input_name: x.shape}
dtype_dict = {k: v.dtype for k, v in params.items()}
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, target,
                                             shape=shape_dict,
                                             dtype=dtype_dict,
                                             params=params)

######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
ctx = tvm.cpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
output_shape = (1, 1000)
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()

def run():
    m.run()

n = 100
print(timeit.timeit('run()', globals=globals(), number=n) / n * 1000, 'msec')
