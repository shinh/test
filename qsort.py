def qsort(a):
    if a:
        v = a[0]
        return (qsort([x for x in a if x < v]) +
                [x for x in a if x == v] +
                qsort([x for x in a if x > v]))
    else:
        return []

a = [1,4,2,4,6,7,1,7,6,1,4,7,8,9,3,1]
print qsort(a)

