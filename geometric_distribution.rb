# https://en.wikipedia.org/wiki/Geometric_distribution
def geometric_distribution(p)
  n = 1
  while rand > p
    n += 1
  end
  n
end

1.upto(10) do |m|
  p = 1.0 / m
  n = 1000
  sum = 0
  1000.times do
    sum += geometric_distribution(p)
  end
  avg = sum.to_f / n
  puts "#{m}: avg=#{avg}"
end
