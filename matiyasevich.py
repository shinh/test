import sys

from z3 import *

chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z = Ints(' '.join(chars))

solver = Solver()

solver.add(w*z + h + j - q == 0)
solver.add((g*k + 2*g + k + 1) * (h + j) + h - z == 0)
solver.add(16 * (k + 1) ** 3 * (k + 2) * (n + 1) ** 2 + 1 - f ** 2 == 0)
solver.add(2*n + p + q + z - e == 0)
solver.add(e ** 3 * (e + 2) * (a + 1) ** 2 + 1 - o ** 2 == 0)
solver.add((a ** 2 - 1) * y ** 2 + 1 - x ** 2 == 0)
solver.add(16 * r ** 2 * y ** 4 * (a ** 2 - 1) + 1 - u ** 2 == 0)
solver.add(n + l + v - y == 0)
solver.add((a ** 2 - 1) * l ** 2 + 1 - m ** 2 == 0)
solver.add(a*i + k + 1 - l - i == 0)
solver.add(((a + u ** 2 * (u ** 2 - a)) ** 2 - 1) * (n + 4*d*y)** 2 + 1 - (x + c*u) ** 2 == 0)
solver.add(p + l * (a - n - 1) + b * (2 * a * n + 2 * a - n**2 - 2*n - 2) - m == 0)
solver.add(q + y * (a - p - 1) + s * (2*a*p + 2*a - p**2 - 2*p - 2) - x == 0)
solver.add(z + p * l * (a - p) + t * (2*a*p - p**2 - 1) - p*m == 0)
solver.add(k == 1)

r = solver.check()
if r == sat:
    print 'sat'
else:
    print r
    sys.exit(1)
