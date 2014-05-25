#!/usr/bin/env ruby

def flush(out)
  if out.size > 0
    puts "echo -en '#{out}'"
  end
end

out = ''
File.read(ARGV[0]).each_byte{|b|
  if out.size > 4000
    flush(out)
    out = ''
  end
  if b >= 32 && b < 127 && b != ?'.ord && b != ?\\.ord
    out += "%c" % b
  else
    out += "\\x%02x" % b
  end
}
flush(out)
