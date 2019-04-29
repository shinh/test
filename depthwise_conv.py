import time

import numpy as np
import chainer
import chainer.links as L

class DepthwiseConv(chainer.Chain):

    def __init__(self):
        super(DepthwiseConv, self).__init__()

        ch = 244
        with self.init_scope():
            self.conv = L.Convolution2D(ch, ch, 3, pad=1, groups=ch)

    def forward(self, x):
        return self.conv(x)


model = DepthwiseConv()
model.to_gpu()
x = model.xp.random.rand(1, 244, 28, 28).astype(np.float32)
print(model(x).shape)
print(model.conv.W.shape)

num_iters = 100
start = time.time()
for i in range(num_iters):
    model(x)
print((time.time() - start) / num_iters * 1000, 'msec')
