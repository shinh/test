def do_lambda
  lambda{
    return true
  }.call
  return false
end

def do_proc
  proc{
    return true
  }.call
  return false
end

def do_times
  3.times{
    return true
  }
  return false
end

p do_proc
p do_lambda
p do_times

__END__

aobjs = []
aobjs.select &filter

def a
  i = 0
  while i < 3
    p lambda {|x|
      return x
    }.call('foo')
    #3.times{
    #  return 1
    #}
    i += 1
    p 'hoge'
  end
end

#def f(g)
#  g[2]
r#nd

#p f(a)
p a


#?\xc8.upto(1e3){|v|p~9*v}

#801.times{|i|p -2000-i*10}
#200.upto(1000){|v|p -10*v}

#?\xc8.upto(1e3){|v|p~9*v}
#-2000.step(-10000,-10) do |v|
#  puts v
#end


