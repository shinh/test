#!/usr/bin/env ruby

require './ctfutils'

file = ARGV[0]

class Exe
  attr_reader :entry

  def initialize(filename)
    @filename = filename
    @code = File.read(filename)
    if @code[0, 4] == "\x7fELF"
      parse_elf
    elsif @code[0, 2] == "MZ"
      parse_pe
    end
  end

  def _get_tmpl(tmpl)
    if @is_64
      tmpl.tr '!', 'Q'
    else
      tmpl.tr '!', 'V'
    end
  end

  def parse_elf
    @is_64 = @code[4].ord == 2
    tmpl = _get_tmpl('vvV!!!Vvvvvvv')
    ehdr = @code[16, 64 - 16].unpack(tmpl)
    @entry = ehdr[3]

    @maps = []
    phoff = ehdr[4]
    phnum = ehdr[9]
    off = phoff
    phnum.times{
      if @is_64
        phdr = @code[off, 56].unpack('VVQQQQQQ')
        p_offset = phdr[2]
        p_vaddr = phdr[3]
        p_filesz = phdr[5]
        p_memsz = phdr[6]
        off += 56
      else
        phdr = @code[off, 32].unpack('VVVVVVVV')
        p_offset = phdr[1]
        p_vaddr = phdr[2]
        p_filesz = phdr[4]
        p_memsz = phdr[5]
        off += 32
      end
      p_type = phdr[0]

      if p_type == 1
        @maps << {
          :offset => p_offset,
          :vaddr => p_vaddr,
          :memsz => p_memsz,
          :filesz => p_filesz,
        }
      end
    }
  end

  def parse_pe
    @maps = []
    `objdump -h #{@filename}`.scan(/\n\s+\d+\s+(\.[a-z]+)(.*)/) do
      #puts "#$1 #$2"
      size, vma, lma, off, algn = $2.split
      @maps << {
        :offset => off.hex,
        :vaddr => vma.hex,
        :memsz => size.hex,
        :filesz => size.hex,
      }
    end
  end

  def get_data(addr)
    if !@maps
      return nil
    end

    @maps.each do |m|
      if addr >= m[:vaddr] && addr < m[:vaddr] + m[:memsz]
        a = addr - m[:vaddr]
        if a >= m[:filesz]
          return '.bss'
        else
          r = []
          o = m[:offset] + a
          c = @code[o].ord

          ok = false
          if c == 10 || c >= 32 && c < 127
            i = @code.index("\0", o)
            s = @code[o..i-1]
            if s =~ /^[ -~\n]+$/
              ok = s.size > 2
              r << s.inspect
            end
          end

          if !ok
            r << '%08x' % @code[o, 4].unpack('V')
            r << '%x' % (@code[o, 8].unpack('Q')[0] >> 32)
          end
          return r * ' '
        end
      end
    end
    return nil
  end

end

exe = Exe.new(file)

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

if `file #{file}` =~ /PE32/
  cmd = "ruby #{File.dirname(File.realpath(__FILE__))}/dispe.rb #{file}"
else
  cmd = "objdump -S #{file}"
end
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
    if !labels[addr] || labels[addr] =~ /^\[L/
      label = "[func#{fid}]"
      labels[addr] = label
      fid += 1
    end
  elsif line =~ /^\s*([0-9a-f]+):\s*.*?\s(callq?|j[a-z]+)\s+(?:0x)?([0-9a-f]+)/
    from = $1.hex
    to = $3.hex
    annots[from] = to
    if !labels[to]
      label = "[L#{lid}]"
      labels[to] = label
      lid += 1
    end
  end
end

if exe.entry
  labels[exe.entry] = '[entry]'
end

dump.each_line do |line|
  addr = line.hex

  if addr != 0 && line !~ /<.*>:/ && (label = labels[addr].to_s) =~ /^\[/
    if label =~ /func/
      puts
    end
    puts "#{label}:"
  end

  annot = []
  if annots[addr]
    label = labels[annots[addr]]
    if !line.include?(label)
      annot << label
    end
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
      data = exe.get_data(addr)
      if data
        annot << data
      end
    end
  end

  line.chomp!
  if !annot.empty?
    line += " ; #{annot * ' '}"
  end
  puts line
end
