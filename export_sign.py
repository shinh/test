#!/usr/bin/env python3

import argparse
import chainer
import chainer.functions as F
import numpy as np
import onnx
import onnx_chainer
from onnx_chainer.replace_func import as_funcnode


class Sign(chainer.Chain):
    def forward(self, x):
        y = F.relu(x)
        y = F.sign(y)
        y = F.relu(y)
        return y


F.sign = as_funcnode('Sign')(F.sign)
def convert_sign(param):
    return onnx.helper.make_node(
        'Sign', param.input_names, param.output_names),
external_converters = {'Sign': convert_sign}

model = Sign()
onnx_chainer.export_testcase(model, [np.array(3.14)], 'sign',
                             external_converters=external_converters)
