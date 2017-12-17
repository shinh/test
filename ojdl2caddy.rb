#!/usr/bin/env ruby

mode = 'wait'
IO.popen("oj dl #{ARGV[0]} 2>&1", 'r').each do |line|
  if mode == 'wait'
    if line =~ /^\[x\] ((in|out)put):/
      mode = $1
    end
  else
    if line =~ /^\[.\] /
      puts
      puts mode == 'input' ? '__INPUT__' : '__OUTPUT__'
      mode = 'wait'
    else
      puts line
    end
  end
end
