N = 6

$strat = []

N.times do |i|
  if i < N/2
    $strat << [*0...N/2]
  else
    $strat << [*N/2...N]
  end
end

def strat(i)
  return $strat[i]

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


10000.times do
  cnt = 0
  ok_cnt = 0
  [*0...N].permutation.each do |perm|
    ok = check(perm)
    if ok
      ok_cnt += 1
    end
    cnt += 1
    # p [perm, ok]
  end

  percent = ok_cnt.to_f * 100 / cnt
  puts "#{ok_cnt}/#{cnt} #{percent}%"

  #if percent > 5
  #  raise
  #end

  tbl = []
  loop {
    cnts = {}
    N.times do |i|
      cnts[i] = N / 2
    end

    tbl = []
    N.times do |i|
      if cnts.size < N/2
        break
      end

      s = []
      (N/2).times do
        r = rand(cnts.size)
        v = cnts.keys[r]
        cnts[v] -= 1
        if cnts[v] == 0
          cnts.delete(v)
        end
        s << v
      end
      tbl << s
    end
    break if tbl.size == N
  }

  $strat = tbl
end
