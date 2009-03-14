require 'benchmark'
a=[]
m={}
5000.times{
  r=rand
  a << r
  m[r] = 1
}

Benchmark.bm{|b|
  b.report() {
    a.each{|v|
      m[v]
    }
  }
}

