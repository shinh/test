#!/usr/bin/python3

import math
import sys

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np
import onnx_chainer

# ic*2==oc
# ic==oc*2
# d2
# g2
# gh
# dw


def non_channel_flops(params):
    iw = params['width']
    k = params['ksize']
    s = params['stride']
    g = params['groups']
    #d = params['dilate']
    assert iw % s == 0
    ow = iw // s
    return ow * ow * k * k / g


def near_even(ch):
    ch = int(ch)
    if ch % 2 == 0:
        return ch
    else:
        return ch + 1


def decide_channel(sqch, g):
    ch = math.sqrt(sqch)
    if g == 1:
        return round(ch)

    if g == 2:
        return near_even(ch)

    assert False


def gen(name, target_mflops, params):
    iw = params['width']
    k = params['ksize']
    s = params['stride']
    g = params['groups']
    d = params['dilate']

    nc_flops = non_channel_flops(params)
    sqch = target_mflops * 1000 * 1000 / nc_flops

    if 'ichan2' in params:
        oc = decide_channel(sqch / 2, g)
        ic = oc * 2
    elif 'ochan2' in params:
        oc = decide_channel(sqch / 2, g)
        ic = oc * 2
    elif 'dwise' in params:
        ic = round(sqch)
        oc = ic
        g = ic
    elif 'dwise2' in params:
        ic = near_even(sqch / 2)
        oc = ic
        g = ic // 2
    else:
        ic = decide_channel(sqch, g)
        oc = ic

    p = (k // 2) * d

    print(ic, oc, g)
    conv = L.Convolution2D(ic, oc, k, s, p, nobias=True,
                           dilate=d, groups=g)
    x = np.random.rand(1, ic, iw, iw).astype(np.float32)

    name = 'conv_%dmflops_%d_%s' % (target_mflops, iw, name)
    onnx_chainer.export_testcase(conv, x, name)


def main():
    target_mflops = int(sys.argv[1])

    for width in [224, 168, 112, 56, 28, 14, 7]:
        base_conv_params = {
            'width': width,
            'ksize': 1,
            'stride': 1,
            'groups': 1,
            'dilate': 1,
        }

        def params(diff):
            p = base_conv_params.copy()
            p.update(diff)
            return p

        gen('1x1', target_mflops, params({}))
        gen('3x3', target_mflops, params({'ksize': 3}))
        gen('5x5', target_mflops, params({'ksize': 5}))
        gen('7x7', target_mflops, params({'ksize': 7}))
        gen('ichan2', target_mflops, params({'ichan2': True}))
        gen('ochan2', target_mflops, params({'ochan2': True}))

        nc_flops = non_channel_flops(params({}))
        sqch = target_mflops * 1000 * 1000 / nc_flops
        if sqch < 500:
            gen('dwise', target_mflops, params({'dwise': True}))
            gen('dwise2', target_mflops, params({'dwise2': True}))
        gen('groups2', target_mflops, params({'groups': 2}))
        if width != 7:
            gen('stride2', target_mflops, params({'stride': 2}))
        gen('dilate2', target_mflops, params({'dilate2': 2}))


main()
