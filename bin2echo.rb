#!/usr/bin/env ruby

$out_filename = ARGV[1]
$is_first = true

def flush(out)
  if out.size > 0
    cmd = "echo -en '#{out}'"
    if $out_filename
      cmd += ' >' if !$is_first
      cmd += "> #{$out_filename}"
      $is_first = false
    end
    puts cmd
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
