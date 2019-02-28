import sys
import timeit

import numpy as np
import onnx
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model

model = onnx.load(sys.argv[1])

ng_func = import_onnx_model(model)
#print(ng_model)

picture = np.ones([1, 3, 224, 224], dtype=np.float32)

runtime = ng.runtime(backend_name='CPU')
#runtime = ng.runtime(backend_name='GPU')
resnet = runtime.computation(ng_func)
#print(resnet)

def run():
  resnet(picture)

n = 100

print(timeit.timeit('run()', globals=globals(), number=n) / n * 1000, 'msec')
