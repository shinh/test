class Array
  def step(n, &pr)
    i = 0
    while true
      a = self[i, n]
      break if a.size != n
      i += n
      pr[*a]
    end
  end
end

File.read('log').split.each_slice(6) do |s1, s2, n1, n2, n3, n4|
  puts "#{n1}"
end
