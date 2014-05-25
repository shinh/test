#!/usr/bin/env ruby

print %Q[#include <unistd.h>
int main(){write(1,"]

prev_escape = false
content = File.read(ARGV[0])
content.each_byte{|b|
  if b >= 32 && b < 127 && b != ?".ord && b != ?\\.ord
    if b.chr =~ /[0-9a-fA-F]/ && prev_escape
      print "\\x%02x" % b
      prev_escape = true
    else
      print "%c" % b
      prev_escape = false
    end
  elsif b == 0
    print "\\0"
    prev_escape = true
  else
    print "\\x%02x" % b
    prev_escape = true
  end
}
puts %Q[",#{content.size});}]
