def global_func
  l = 2
  [47, self]
end

class C
  def call_jit_func(jit_func)
    l = 3
    #r = jit_func.call
    r = []
    r + [get_iv, self, l]
  end

  def get_iv
    @v
  end
end

class D
  def test_call_jit_func
    c = C.new
    jit_method = c.method(:call_jit_func)
    jit_func = method(:global_func)
    puts "jit_method recv: #{jit_method.receiver}"
    puts "jit_func recv: #{jit_func.receiver}"
    jit_method.call(jit_func)
  end
end

p D.new.test_call_jit_func

#C.new.mf.call
#C.new.send(:gf)
