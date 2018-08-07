import numpy as np
import chainer
import tensorflow as tf

# ONNX's
def softmax_2d(x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

F = chainer.functions
a = np.array([[1.,2.],[2.,5.]])
a = np.array([[[1.,2.],[2.,5.]],[[1.,2.],[2.,5.]]])

print(F.softmax(a))
print(tf.Session().run(tf.nn.softmax(a)))

print(F.softmax(a.reshape(2, 4)))
print(softmax_2d(a.reshape(2, 4)).reshape(2, 2, 2))

