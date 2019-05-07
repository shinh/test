import time
import numpy as np
import cupy
import sys

import chainer
import chainer.functions as F

if len(sys.argv) > 1:
    chainer.config.use_cudnn = sys.argv[1]

xp = cupy

bs = 32
ch = 1024

x = xp.random.rand(bs, ch, 7, 7)
w = xp.random.rand(ch, 1, 3, 3)

assert (bs, ch, 7, 7) == F.convolution_2d(x, w, pad=1, groups=ch).shape

start = time.time()
count = 0
while True:
    F.convolution_2d(x, w, pad=1, groups=ch)
    cupy.cuda.device.Device().synchronize()

    count += 1
    now = time.time()
    if now > start + 1:
        break

print('%.3f msec (%d times)' % ((now - start) / count * 1000, count))
