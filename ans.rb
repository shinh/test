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
      x = ceil_div((x + 1) * l, c0) - 1
    else
      x = x * l / c1
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
    z = ceil_div(x * c1, l)
    s = ceil_div((x + 1) * c1, l) - z
    o += s.to_s
    if s == 0
      x = x - z
    else
      x = z
    end
  end
  o.reverse
end

def test_uabs(l, r)
  o = rand_bin(l, r)
  e = enc_uabs(o)
  d = dec_uabs(*e)
  cl = e[0].to_s(2).size
  tl = e[1,2].map{|c|
    r = c.to_f / l
    -c * Math.log2(r)
  }.sum.to_i
  puts "uABS orig_bits=#{l} pr(0)=#{r} cmp_bits=#{cl} shannon=#{tl}"
  puts o == d ? "OK" : "FAIL"
end

test_uabs(800, 0.1)
test_uabs(800, 0.3)
test_uabs(800, 0.5)

def rand_str(cs)
  str = []
  cs.each_with_index do |c, i|
    str += [i] * c
  end
  str.shuffle!

  if cs.size <= 36
    str.map{|i|i.to_s(36)}
  else
    str
  end
end

def cumsum(f)
  s = 0
  cdf = []
  f.each do |c|
    cdf << s
    s += c
  end
  raise if s != f.sum
  cdf
end

def quantized_distribution(hist, l, n)
  f = []
  hist.each do |sym, cnt|
    f << [cnt * 2 ** n / l, 1].max
  end

  adj = 2 ** n - f.sum
  mi = hist.each_with_index.map{|t, i|[t[1], i]}.max[1]
  f[mi] += adj
  raise if f.sum != 2 ** n
  f
end

def get_hist(str)
  hist = {}
  str.each do |sym|
    hist[sym] ||= 0
    hist[sym] += 1
  end
  hist
end

def get_shannon_limit(str)
  s = 0
  get_hist(str).each do |_, cnt|
    s += -cnt * Math.log2(cnt.to_f / str.size)
  end
  s.to_i
end

def enc_rans(str)
  hist = get_hist(str)

  l = str.size
  n = 12
  f = quantized_distribution(hist, l, n)
  cdf = cumsum(f)

  s2i = {}
  hist.each_with_index do |t, i|
    s2i[t[0]] = i
  end

  x = 0
  str.each do |sym|
    s = s2i[sym]
    fs = f[s]
    x = ((x / fs) << n) + (x % fs) + cdf[s]
  end

  return x, l, n, f, cdf, s2i
end

def dec_rans(x, l, n, f, cdf, s2i)
  i2s = {}
  s2i.each do |s, i|
    i2s[i] = s
  end

  mask = 2 ** n - 1

  symbol = []
  f.each_with_index do |c, i|
    c.times do
      symbol << i
    end
  end
  raise if symbol.size != 2 ** n

  o = []
  l.times do
    s = symbol[x & mask]
    o << i2s[s]
    x = f[s] * (x >> n) + (x & mask) - cdf[s]
  end
  return o.reverse
end

def test_rans(o)
  e = enc_rans(o)
  d = dec_rans(*e)
  cl = e[0].to_s(2).size
  tl = get_shannon_limit(o)
  puts "rANS orig_syms=#{o.size} cmp_bits=#{cl} shannon=#{tl}"
  puts o == d ? "OK" : "FAIL"
end

test_rans(rand_str((0..15).map{|i|10}))
test_rans(rand_str((0..15).map{|i|(i + 1) * 10}))
test_rans(rand_str((0..15).map{|i|i * i + 1}))
test_rans(rand_str((0..127).map{|i|(i + 1)}))
