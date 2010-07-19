n = ARGV[0].to_i
fib = [1,1]
n.times{|i|
  fib << fib[i+1] + fib[i]
}
