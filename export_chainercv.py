#!/usr/bin/env python

import argparse
import chainer
import numpy as np

import chainercv.links as C
import onnx_chainer

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--model', default='ResNet50')
parser.add_argument('--kwargs', default='{}')
args = parser.parse_args()

model = getattr(C, args.model)(**eval(args.kwargs))
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

# Pseudo input
x = model.xp.zeros((1, 3, args.size, args.size), dtype=np.float32)

onnx_chainer.export_testcase(model, x, '%s_%d' % (args.model, args.size))
