#!/usr/bin/env ruby

$<.each do |log|
  log.gsub!(/\a                        # Bell
           | \e \x5B .*? [\x40-\x7E]  # CSI
           | \e \x5D .*? \x07         # Set terminal title
           | \e [\x40-\x5A\x5C\x5F]   # 2 byte sequence
            /x, '')
  log.gsub!(/\s* \x0d* \x0a/x, "\x0a")  # Remove end-of-line CRs.
  log.gsub!(/ \s* \x0d /x, "\x0a")      # Replace orphan CRs with LFs.

  print log
end
