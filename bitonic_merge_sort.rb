def bitonic_sort(a, b, e, z)
  s = (e-b)/2
  return if s == 0
  s.times{|i|
    if a[b+i+s] && ((z != 0) ^ (a[b+i] > a[b+i+s]))
      a[b+i], a[b+i+s] = a[b+i+s], a[b+i]
    end
  }
  bitonic_sort(a, b, b + s, z)
  bitonic_sort(a, b + s, e, z)
end

def bitonic_merge_sort(a)
  s = 1
  n = a.size
  while n > 1
    n /= 2
    s *= 2
    n.times{|i|
      bitonic_sort(a, s*i, s*i+s, i%2)
    }
  end
end

a = [*0...32].sort_by{rand}
p a
bitonic_merge_sort(a)
p a

