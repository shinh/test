import time

import numpy as np
import cupy
from cupy import cutensor

a = cupy.random.rand(1024, 1, 9).astype(np.float32)
b = cupy.random.rand(1024, 9, 1568).astype(np.float32)

e = cupy.matmul(a, b)

c = cupy.zeros((1024, 1, 1568)).astype(np.float32)

desc_a = cutensor.create_tensor_descriptor(a)
desc_b = cutensor.create_tensor_descriptor(b)
desc_c = cutensor.create_tensor_descriptor(c)

mode_a = ('i', 'j', 'l')
mode_b = ('i', 'l', 'k')
mode_c = ('i', 'j', 'k')

def my_matmul(a, b):
    return cutensor.contraction(1.0, a, desc_a, mode_a,
                                b, desc_b, mode_b,
                                0.0, c, desc_c, mode_c)

c = my_matmul(a, b)

cupy.testing.assert_allclose(e, c, rtol=1e-6)

for fn in [cupy.matmul, my_matmul]:
    print(fn)

    start = time.time()
    count = 0
    while True:
        fn(a, b)
        cupy.cuda.device.Device().synchronize()
        count += 1
        now = time.time()
        if now > start + 1:
            break

    print('%.3f msec (%d times)' % ((now - start) / count * 1000, count))
