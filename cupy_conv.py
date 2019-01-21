import numpy as np
import cupy
from cupy.cuda import cudnn
import chainer


def aranges(*shape):
    r = np.prod(shape)
    return np.arange(r).reshape(shape).astype(np.float32)


bsize = 2
ichan = 4
ochan = 5
iw = 9
ih = 9
kw = 3
kh = 3
ow = iw - kw + 1
oh = ih - kh + 1


def conv(x, w, b, x_nhwc=False, w_nhwc=False):
    y_shape = (bsize, ochan, ow, oh)
    d_layout = cudnn.CUDNN_TENSOR_NCHW
    w_layout = cudnn.CUDNN_TENSOR_NCHW

    if x_nhwc:
        d_layout = cudnn.CUDNN_TENSOR_NHWC
        x = np.transpose(x, (0, 2, 3, 1))
        y_shape = (bsize, ow, oh, ochan)

    if w_nhwc:
        w_layout = cudnn.CUDNN_TENSOR_NHWC
        w = np.transpose(w, (0, 2, 3, 1))

    x = cupy.array(x)
    w = cupy.array(w)
    b = cupy.array(b)
    y = cupy.ones(y_shape, dtype=np.float32)
    cupy.cudnn.convolution_forward(x, w, b, y, (0, 0), (1, 1), (1, 1), 1,
                                   auto_tune=True, tensor_core='auto',
                                   d_layout=d_layout, w_layout=w_layout)

    y = chainer.cuda.to_cpu(y)
    if x_nhwc:
        y = np.transpose(y, (0, 3, 1, 2))
    return y


x = aranges(bsize, ichan, ih, iw)
w = aranges(ochan, ichan, kh, kw)
b = aranges(ochan)
print(conv(x, w, b).shape)
print(conv(x, w, b, x_nhwc=True).shape)
print(np.allclose(conv(x, w, b), conv(x, w, b, x_nhwc=True)))

#print(conv(x, w, b, w_nhwc=True).shape)
print(conv(x, w, b, x_nhwc=True, w_nhwc=True).shape)
