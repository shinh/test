# https://docs.tvm.ai/tutorials/language/reduction.html#sphx-glr-tutorials-language-reduction-py

from __future__ import absolute_import, print_function

import tvm
import numpy as np


n = tvm.var("n")
m = tvm.var("m")
A = tvm.placeholder((n, m), name='A')
k = tvm.reduce_axis((0, m), "k")
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name="B")

s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))

s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))

s = tvm.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))

xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
s[B].bind(xi, tvm.thread_axis("threadIdx.y"))
tx = tvm.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], "cuda")
print(fcuda.imported_modules[0].get_source())

nn = 128
ctx  = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), ctx)
fcuda(a, b)
tvm.testing.assert_allclose(
    b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)
