#!/usr/bin/env ruby

$<.each do |line|
  line.gsub!(/\a                        # Bell
             | \e \x5B .*? [\x40-\x7E]  # CSI
             | \e \x5D .*? \x07         # Set terminal title
             | \e [\x40-\x5A\x5C\x5F]   # 2 byte sequence
             /x, '')
  line.gsub!(/\s* \x0d* \x0a/x, "\x0a")  # Remove end-of-line CRs.
  line.gsub!(/ \s* \x0d /x, "\x0a")      # Replace orphan CRs with LFs.
  puts line
end
