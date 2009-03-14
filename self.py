class C:
    def f():
        print 'hoge'
    def g(self):
        self.f()  # TypeError: f() takes no arguments (1 given)
C().g()
