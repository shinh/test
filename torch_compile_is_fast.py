import time
import torch

N = 1000

def func(x):
    a = torch.ones(N, N)
    b = torch.ones(N, N)
    return x + (a @ b).sum(1)

def bench(fn):
    x = torch.ones(N)
    fn(x)
    start = time.time()
    count = 0
    while start + 1 > time.time():
        fn(x)
        count += 1
    return count

base = bench(func)
opt = bench(torch.compile(func, mode="max-autotune"))
print(opt / base, "x speed up!!!")
