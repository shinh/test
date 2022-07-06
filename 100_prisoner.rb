N = 6

def strat(i)
  return [
    [0,1,2],
    [0,1,2],
    [0,1,2],
    [3,4,5],
    [3,4,5],
    [3,4,5],
  ][i]

  r = []
  n = N/2
  n.times do |j|
    r << (j * 4 + i) % N
  end
  r

  # if i < N/2
  #   [*0...N/2]
  # else
  #   [*N/2...N]
  # end
end

def check(perm)
  perm.each_with_index do |v, i|
    if !strat(i).include?(v)
      return false
    end
  end
  return true
end

cnt = 0
ok_cnt = 0
[*0...N].permutation.each do |perm|
  ok = check(perm)
  if ok
    ok_cnt += 1
  end
  cnt += 1
  p [perm, ok]
end

puts "#{ok_cnt}/#{cnt} #{ok_cnt.to_f * 100 / cnt}%"

