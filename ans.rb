# https://en.wikipedia.org/wiki/Asymmetric_numeral_systems

def rand_bin(n, r)
  s = ''
  n.times do
    s += rand < r ? '0' : '1'
  end
  s
end

def ceil_div(a, b)
  (a + b - 1) / b
end

def enc_uabs(bin)
  # if s = 0 then new_x = ceil((x+1)/(1-p)) - 1 // C(x,0) = new_x
  # if s = 1 then new_x = floor(x/p)  // C(x,1) = new_x
  c0 = bin.count('0')
  c1 = bin.count('1')
  l = c0 + c1
  x = 0
  bin.each_char do |s|
    if s == '0'
      x = ceil_div((x + 1) * l, c1) - 1
    else
      x = x * l / c0
    end
  end
  [x, c0, c1]
end

def dec_uabs(e, c0, c1)
  # s = ceil((x+1)*p) - ceil(x*p)  // 0 if fract(x*p) < 1-p, else 1
  # if s = 0 then new_x = x - ceil(x*p)   // D(x) = (new_x, 0)
  # if s = 1 then new_x = ceil(x*p)  // D(x) = (new_x, 1)
  l = c0 + c1
  x = e
  o = ''
  l.times do
    s = ceil_div((x + 1) * c0, l) - ceil_div(x * c0, l)
    o += s.to_s
    if s == 0
      x = x - ceil_div(x * c0, l)
    else
      x = ceil_div(x * c0, l)
    end
  end
  o.reverse
end

o = rand_bin(800, 0.3)
e = enc_uabs(o)
d = dec_uabs(*e)
puts o == d ? "uABS OK" : "uABS FAIL"
