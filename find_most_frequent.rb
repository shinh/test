#!/usr/bin/env ruby

def gen
  r = ""
  23.times do
    r += "abc"[rand(3)]
  end
  r
end

def solve(i)
  l = []
  l << [i.count("a"), "a"]
  l << [i.count("b"), "b"]
  l << [i.count("c"), "c"]
  l.sort!
  if l[-2][0] == l[-1][0]
    return nil
  end
  return l[-1][1] * l[-1][0]
end

100.times do
  i = gen
  e = solve(i)
  next if !e
  o = `echo #{i} | sed -f find_most_frequent.sed`.split("\n")[-1]
  if o != e
    puts "#{i} #{o}"
  end
end
