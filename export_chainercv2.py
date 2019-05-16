#!/usr/bin/python3

import argparse
import sys

import chainer
import numpy as np
import onnx_chainer
from chainercv2.model_provider import get_model as chcv2_get_model


parser = argparse.ArgumentParser(description='Export ChainerCV2 model')
parser.add_argument('name')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--shape', default='1,3,224,224')
parser.add_argument('--image', type=str)

args = parser.parse_args()

model = chcv2_get_model(args.name, pretrained=True)
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

shape = tuple(map(int, args.shape.split(',')))
if args.image is None:
    x = model.xp.random.rand(*shape).astype(model.xp.float32)
else:
    from PIL import Image
    img = Image.open(args.image)
    img = img.resize(shape[2:])
    x = model.xp.array(np.asarray(img))
    x = x / 256.0
    x = x.astype(model.xp.float32)
    x = x.reshape(shape)

sizestr = 'x'.join(map(str, shape[2:]))
onnx_chainer.export_testcase(model, x, '%s_%s' % (args.name, sizestr))
