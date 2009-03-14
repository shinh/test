import imp
import sys

class Hook(object):
    def _get_code(self, fullname):
        f = file(fullname)
        return False, f.read()

    def load_module(self, fullname):
        print fullname

        ispkg, code = self._get_code(fullname)

        mod = sys.modules.setdefault(fullname, imp.new_module(fullname))
        mod.__file__ = "<%s>" % self.__class__.__name__
        mod.__loader__ = self.__class__
        #mod.__loader__ = None
        #print mod.__dict__
        exec code in mod.__dict__
        #exec code in {'__loader__': self}
        #exec code in {'__loader__': None, '__name__': 'code.py'}

        sys.exit(0)

        mod = sys.modules.setdefault(fullname, imp.new_module(fullname))
        mod.__file__ = "<%s>" % self.__class__.__name__
        mod.__loader__ = self
        if ispkg:
            mod.__path__ = []
        exec code in mod.__dict__
        return mod

hook = Hook()
hook.load_module('hoge.py')
