#!/usr/bin/env ruby
if !c=ARGV[0]
  raise "#$0 <pid or process name>"
end
t=`ps -auxww`.map do |l|
  a=l.split
  [a[1].to_i, a[10]]
end.select do |i,n|
  if c=~/^\d+$/
    i==c.to_i
  else
    n.index(c)
  end
end
if t.size>1&&/^\d+$/!~c
  u=t.select do |i,n|
    n=~/(^|\/)#{c}$/
  end
  t=u if u.size==1
end
if t.empty?
  puts"No match"
elsif t.size>1
  puts"Umbiguous:"
  t.each{|i,n|puts"#{i} #{n}"}
else
  #open('/tmp/bt','w'){|of|of.puts('bt')}
  open('/tmp/bt','w'){|of|of.puts('thread apply all bt')}
  exec"gdb -nx -x /tmp/bt -batch -q -p #{t[0][0]}"
end
