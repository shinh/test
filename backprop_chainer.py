import numpy as np
import chainer
from chainer import functions as F

a = chainer.Variable(np.array(6.))
b = chainer.Variable(np.array(4.))
c = chainer.Variable(np.array(2.))

r = (a + b) * (F.log(b) + c)
# r = (a + b) * (F.log(np.array(4.)) + c)

r.grad = np.array(5.0)
r.backward()
print(a.grad, b.grad, c.grad)
