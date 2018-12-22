import numpy as np
import cupy
import chainerx as chx
import sys
import timeit

def cupy_ones(*args, **kwargs):
    return cupy.ones(*args, **kwargs)

def chx_ones(*args, **kwargs):
    return chx.ones(*args, **kwargs, device='cuda:0')

def bench(ones, cnt):
    arrays = []
    for i in range(cnt):
        a = ones((25, 1000, 1000), np.float32)
        arrays.append(a)

    x = arrays[0]
    y = arrays[1]
    if 'warmup' in sys.argv:
      x * y
    def run():
        r = x * y
        cupy.cuda.device.Device().synchronize()
        return r
    print(timeit.timeit('run()', number=100, globals=locals()))


if 'managed' in sys.argv:
    cupy.cuda.set_allocator(
        cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
if 'cupy' in sys.argv:
    bench(cupy_ones, 3)
else:
    bench(chx_ones, 3)
