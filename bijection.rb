def f(x, y)
  (x + y) * (x + y + 1) / 2 + x
end

s = {}
10.times{|a|
  10.times{|b|
    10.times{|c|
      10.times{|d|
        v = f(a, f(b, f(c, d)))
        s[v] = true
      }
    }
  }
}

puts s.size
