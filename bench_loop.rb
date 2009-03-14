require 'benchmark'
N=10000000
Benchmark.bm do |b|
  b.report('for:   '){
    s = 0
    for i in 0...N
      s += i
    end
    p s
  }
  b.report('times: '){
    s = 0
    N.times do |i|
      s += i
    end
    p s
  }
  b.report('upto:  '){
    s = 0
    0.upto(N-1) do |i|
      s += i
    end
    p s
  }
  b.report('while: '){
    s = 0
    i = 0
    while i <= N
      s += i
      i += 1
    end
    p s
  }
end
