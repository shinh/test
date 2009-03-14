def f(a=[]):
    a.append(1)
    return a

print f()
print f()  # [1, 1]
