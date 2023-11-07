# https://core.ac.uk/download/pdf/82128039.pdf

import numpy as np

a = np.arange(6 * 4).reshape(6, 4)

m1 = 3
m2 = 2
n1 = 2
n2 = 2
ra = a.reshape(m1, m2, n1, n2).transpose(2, 0, 3, 1).reshape(m1 * n1, m2 * n2)

u, s, v = np.linalg.svd(ra)

#print(u[:, :4] @ np.diag(s) @ v)

s1 = np.sqrt(s[0])

b = s1 * u[:, 0]
c = s1 * v[0, :]
b = b.reshape(n1, m1).T
c = c.reshape(n2, m2).T
print(np.kron(b, c))
