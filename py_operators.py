import operator


ops = set()
for o in dir(operator):
    ops.add(o)

for o in sorted(ops):
    if o.startswith('_'):
        continue
    if o.endswith('_'):
        o = o[:-1]
    o = '__%s__' % o
    if o in ops:
        print('    def %s(self, *args):' % o)
        print("        return self.__getattr__('%s')(*args)" % o)
        print('')
