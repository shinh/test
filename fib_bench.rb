#!/usr/bin/env ruby

TYPE = 'cc'

CMD = {
  'rb' => '~/src/ruby/ruby-1.9 fib.rb %d',
  'hs' => './a.out %d +RTS -K10M',
  'java' => 'java fib %d',
  'cc' => './a.out %d',
}

#v = (1..13).map{|i|i * 10000}
v = (1..20).map{|i|i * 10000} + [220000]
#v = (1..20).map{|i|i * 10000} + [220000,250000,300000,350000,400000]
rss = []
time = []
v.each do |i|
  STDERR.puts i
  cmd = CMD[TYPE] % i
  r = `/usr/bin/time -v #{cmd} 2>&1`
  rss << r[/Maximum resident set size.*?: (\d+)/, 1]
  time << r[/User time .*?: (.*)/, 1]
end

[[rss, 'fib_rss_%s.dat'], [time, 'fib_time_%s.dat']].each do |n, f|
  File.open(f % TYPE, 'w') do |of|
    v.zip(n).each do |k, a|
      of.puts "#{k} #{a}"
    end
  end
end
