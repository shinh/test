#!/usr/bin/env ruby

v = (1..20).map{|i|i * 10000} + [220000,250000,300000,350000,400000]
rss = []
time = []
v.each do |i|
  STDERR.puts i
  r = `/usr/bin/time -v ./a.out #{i} +RTS -K10M 2>&1`
  rss << r[/Maximum resident set size.*?: (\d+)/, 1]
  time << r[/User time .*?: (.*)/, 1]
end

[[rss, 'fib_rss.dat'], [time, 'fib_time.dat']].each do |n, f|
  File.open(f, 'w') do |of|
    v.zip(n).each do |k, a|
      of.puts "#{k} #{a}"
    end
  end
end
