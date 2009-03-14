def strict_list(t):
    class l(list):
        def append(self, v):
            if type(v) is not t:
                raise TypeError('%s expected, but comes %s' % (t, type(v)))
            list.append(self, v)
    return l

l = strict_list(int)()
l.append(3)
l.append([3,4])
