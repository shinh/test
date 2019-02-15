import operator


ops = set()
for o in dir(operator):
    ops.add(o)

oops = []
for o in sorted(ops):
    if o.startswith('_'):
        continue
    if o.endswith('_'):
        o = o[:-1]
    if o.startswith('i') and '__%s__' % o[1:] in ops:
        continue

    o = '__%s__' % o
    if o in ops:
        oops.append(o)

oops.extend(
    ['__radd__', '__rsub__', '__rmul__', '__rdiv__', '__rtruediv__',
     '__rfloordiv__', '__rmod__', '__rdivmod__', '__rpow__', '__rlshift__',
     '__rrshift__', '__rand__', '__rxor__', '__ror__'])

for o in oops:
    print('    def %s(self, *args):' % o)
    print("        return self.__getattr__('%s')(*args)" % o)
    print('')
