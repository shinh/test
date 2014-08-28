#!/usr/bin/env ruby

pe = ARGV[0]

base = nil
is_64 = false
state = []
reloc_map = {}
entry = nil

`objdump -x #{pe}`.each_line do |line|
  if state[0] == :import_table
    if line !~ /^\s/
      state = []
      next
    end

    toks = line.split
    if toks.size == 6 && line =~ /^[ \t0-9a-f]+\s([0-9a-f]+)$/
      state[1] = $1.hex
    elsif toks.size == 3 && toks[0] =~ /^[0-9a-f]+$/
      addr = base + state[1]
      #puts "%x %s" % [addr, toks[2]]
      reloc_map[addr] = toks[2]
      state[1] += is_64 ? 8 : 4
    end
  else
    if line =~ /pei-x86-64/
      is_64 = true
    elsif line =~ /AddressOfEntryPoint\s+([0-9a-f]+)/
      entry = $1.hex
    elsif line =~ /ImageBase\s+([0-9a-f]+)/
      base = $1.hex
    elsif line =~ /The Import Tables /
      state = [:import_table, nil]
    end
  end
end

entry += base

`objdump -S #{pe}`.each_line do |line|
  if line =~ /\s# 0x([0-9a-f]+)$/ || line =~ /call\s+\*0x([0-9a-f]+)$/
    addr = $1.hex
    if name = reloc_map[addr]
      line = line.chomp + " <#{name}>"
    end
  end
  puts line
end
