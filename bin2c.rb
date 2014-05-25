#!/usr/bin/env ruby

out = %Q[#include <unistd.h>
int main(){write(1,"]

prev_escape = false
content = File.read(ARGV[0])
content.each_byte{|b|
  if b >= 32 && b < 127 && b != ?".ord && b != ?\\.ord
    if b.chr =~ /[0-9a-fA-F]/ && prev_escape
      out += "\\x%02x" % b
      prev_escape = true
    else
      out += "%c" % b
      prev_escape = false
    end
  elsif b == 0
    out += "\\0"
    prev_escape = true
  else
    out += "\\x%02x" % b
    prev_escape = true
  end
}
out += %Q[",#{content.size});}]

puts out
