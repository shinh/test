#!/usr/bin/env ruby

file = ARGV[0]

syms = {}
['nm', 'nm -D'].each do |nm|
  `#{nm} #{file}`.each_line do |line|
    toks = line.split
    if toks.size != 3
      next
    end
    addr, type, name = *toks
    syms[addr.hex] = name
  end
end

cmd = "objdump -S #{file}"
puts cmd
dump = `#{cmd}`

annots = {}

fid = 0
lid = 0
labels = {}
dump.each_line do |line|
  if line =~ /^([0-9a-f]+) <(.*)>:$/
    labels[$1.hex] = $2
  elsif line =~ /push\s+%[er]bp/
    addr = line.hex
    if !labels[addr] || labels[addr] =~ /^\.L/
      label = ".func#{fid}"
      labels[addr] = label
      fid += 1
    end
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
  if addr != 0 && (label = labels[addr].to_s) =~ /^\.(L|func)/
    if label =~ /func/
      puts
    end
    puts "#{label}:"
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
      next if operand =~ /</
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