#!/usr/bin/env ruby

def split_toks(line)
  toks = []
  line.scan(/\G((\d+)|[^\d]+)/) do
    if $2
      toks << $2.to_i
    else
      toks << $1
    end
  end
  toks
end

lines = []
$<.each do |line|
  lines << split_toks(line)
end

lines.sort.each do |toks|
  puts toks * ""
end
