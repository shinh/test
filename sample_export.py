#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy

import chainer
import chainercv.links as C
import onnx_chainer

model = C.ResNet50(pretrained_model='imagenet', arch='he')
#model = C.VGG16(pretrained_model='imagenet')
model.to_gpu()

# Pseudo input
x = chainer.Variable(cupy.zeros((1, 3, 224, 224), dtype=np.float32))

onnx_chainer.export(model, x, filename='resnet50.onnx')
