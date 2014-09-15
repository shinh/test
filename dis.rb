#!/usr/bin/env ruby

file = ARGV[0]

syms = {}
`nm #{file}`.each_line do |line|
  toks = line.split
  if toks.size != 3
    next
  end
  addr, type, name = *toks
  syms[addr.hex] = name
end

cmd = "objdump -S #{file}"
puts cmd
dump = `#{cmd}`

annots = {}

lid = 0
labels = {}
dump.each_line do |line|
  if line =~ /^([0-9a-f]+) <(.*)>:$/
    labels[$1.hex] = $2
  elsif line =~ /^\s*([0-9a-f]+):\s*.*?\s(call|j[a-z]+)\s+([0-9a-f]+)/
    from = $1.hex
    to = $3.hex
    if !labels[to]
      label = ".L#{lid}"
      labels[to] = label
      annots[from] = label
      lid += 1
    end
  end
end

dump.each_line do |line|
  addr = line.hex
  if addr != 0 && labels[addr].to_s =~ /^\./
    puts "#{labels[addr]}:"
  end

  annot = []
  if annots[addr]
    annot << annots[addr]
  end

  if line =~ /:\s+([0-9a-f]{2}\s)+\s+([a-z]+)\s+(.*)$/
    op = $2
    operands = $3
    operands.split(',').each do |operand|
      next if operand =~ /^-?0x[0-9a-f]+\(/
      next if operand =~ /^\(?%/
      addr = operand[/(0x)?([0-9a-f]+)/, 2].hex
      if syms[addr]
        annot << syms[addr]
      end
    end
  end

  line.chomp!
  if !annot.empty?
    line += " ; #{annot * ' '}"
  end
  puts line
end
