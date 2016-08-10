#!/usr/bin/env ruby

$: << File.dirname(__FILE__)

require 'bsearch'
require 'ctfutils'

if ARGV[0] == '--ebx-thunk'
  ARGV.shift
  ebx_thunk = ARGV.shift.hex
  STDERR.puts "Use ebx_thunk=%x" % ebx_thunk
end

if ARGV[0] == 'objdump'
  cmd = ARGV * ' '
  file = ARGV[-1]
else
  file = ARGV[0]
end
out_filename = file.sub(/\.exe$/, '') + '.dmp'

comments = {}
cur_comment = []
if File.exist?(out_filename)
  File.readlines(out_filename).each do |line|
    if line =~ /# (.*)/
      comment = $1
      if line =~ /^\s*(\h+):/
        addr = $1.hex
        comments[[addr, true]] = comment
      else
        cur_comment << comment
      end
    end

    if line =~ /^\s*(\h+):/
      addr = $1.hex
      if !cur_comment.empty?
        comments[[addr, false]] = cur_comment
        cur_comment = []
      end
    end
  end
end

class Exe
<<<<<<< 8882648de252f8fbfa1213e9386a10098eae86a9
  attr_reader :entry, :is_pie, :filename
=======
  attr_reader :entry, :is_pie, :is_cgc
>>>>>>> Better suppport for CGC

  def initialize(filename)
    @filename = filename
    @code = File.read(filename)
    if @code[0, 4] == "\x7fELF"
      parse_elf
    elsif @code[0, 4] == "\x7fCGC"
<<<<<<< 8882648de252f8fbfa1213e9386a10098eae86a9
      @code[0, 4] = "\x7fELF"
      @code[7] = "\0"
      @filename = "/tmp/#{File.basename(@filename)}"
      File.write(@filename, @code)
=======
      @is_cgc = true
>>>>>>> Better suppport for CGC
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
    @is_pie = false

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
        if p_vaddr == 0
          @is_pie = true
        end
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
  if exe.is_cgc
    nm = 'i386-linux-cgc-' + nm
  end
  `#{nm} #{file}`.each_line do |line|
    toks = line.split
    if toks.size != 3
      next
    end
    addr, type, name = *toks
    if addr.hex >= 0x100
      syms[addr.hex] = name
    end
  end
end

of = File.open(out_filename, 'w')

if !cmd
  if exe.is_cgc
    cmd = "i386-linux-cgc-objdump -S #{exe.filename}"
  elsif `file #{file}` =~ /PE32/
    cmd = "ruby #{File.dirname(File.realpath(__FILE__))}/dispe.rb #{exe.filename}"
  elsif `file #{file}` =~ / ARM/
    cmd = "arm-linux-gnueabihf-objdump -S #{exe.filename}"
  elsif `file #{file}` =~ / SH,/
    cmd = "/usr/local/stow/binutils-all/bin/all-objdump -S #{exe.filename}"
  else
    cmd = "objdump -S #{exe.filename}"
  end
end
of.puts cmd
dump = `#{cmd} | c++filt`
dump.gsub!(/(\s+\h+\s+)<.+@plt\+0x\h+>/, '\1')

annots = {}

fid = 0
lid = 0
labels = {}
calls = []

if ebx_thunk
  labels[ebx_thunk] = '[func_ebx]'
end
dump.each_line do |line|
  if line =~ /^(\h+) <(.*)>:$/
    labels[$1.hex] = $2
  end
end

dump.each_line do |line|
  if line =~ /^(\h+) <(.*)>:$/
    labels[$1.hex] = $2
  elsif line =~ /push\s+(%[er]bp|{lr})/
    addr = line.hex
    if !labels[addr] || labels[addr] =~ /^\[L/
      label = "[func#{fid}]"
      labels[addr] = label
      fid += 1
    end
  elsif line =~ /^\s*(\h+):\s*.*?\s(callq?|j[a-z]+|b(?:l|lx|sr|t|f))\s+(?:0x)?(\h+)/
    from = $1.hex
    to = $3.hex
    if $2 =~ /^call|^bl/
      is_func = true
    end
    annots[from] = to
    if !labels[to]
      if is_func
        label = "[func#{fid}]"
        labels[to] = label
        fid += 1
      else
        label = "[L#{lid}]"
        labels[to] = label
        lid += 1
      end
    end

    if is_func
      calls << [from, labels[to]]
    end
  end
end

if exe.entry
  labels[exe.entry] = '[entry]'
end

ebx = nil
last_addr = nil
dump.each_line do |line|
  addr = line.hex
  if addr != 0
    last_addr = addr
  end

  if ebx_thunk
    if line =~ /call\s+(0x)?#{"%x"%ebx_thunk}/
      ebx = addr + 5
    elsif line =~ /add\s+\$0x(\h+),%ebx/
      #puts "%x => %x" % [ebx, ebx + $1.hex]
      if ebx
        ebx += $1.hex
      end
    end
  end

  if addr != 0 && line !~ /<.*>:/ && (label = labels[addr].to_s) =~ /^\[/
    if label =~ /func/
      of.puts ""
    end
    of.puts "#{label}:"
  end

  if line =~ /^\s*(\h+):/ && c = comments[[addr, false]]
    of.puts c.map{|l|"# #{l}\n"} * ""
  end

  annot = []
  if annots[addr]
    label = labels[annots[addr]]
    if !line.include?(label)
      annot << label
    end
  end

  if line =~ /:\s+((?:\h{2}\s)+)\s+([a-z]+)\s+(.*)$/
    ip = $`.hex
    num_ops = $1.split.size
    op = $2
    operands = $3
    operands.split(',').each do |operand|
      if operand =~ /(-?0x\h+)\((%rip|%ebx)\)/
        if $2 == '%ebx'
          if ebx_thunk && ebx
            a = ebx + $1.hex
          end
        else
          a = ip + $1.hex + num_ops
        end
      else
        next if exe.is_pie
        next if operand =~ /^-?0x\h+\(/
        next if operand =~ /^\(?%/
        next if operand =~ /</
        next if !operand[/(0x)?(\h+)/, 2]
        a = operand[/(0x)?(\h+)/, 2].hex
      end

      if syms[a]
        annot << syms[a]
      end
      if a
        data = exe.get_data(a)
      end
      if data
        annot << data
      end
    end
  end

  line.chomp!
  if !annot.empty?
    line += " ; #{annot * ' '}"
  end
  if c = comments[[addr, true]]
    line += " # #{c}"
  end
  of.puts line
end

funcs = {}
funcs[nil] = [[], []]
addrs = []
labels_a = labels.to_a.sort.reject{|a, fn|fn =~ /^\[L/}
labels_a.each_with_index do |kv, i|
  addr, fn = kv
  funcs[fn] = [[], []]
  next_addr = labels_a[i+1] ? labels_a[i+1][0] : last_addr
  addrs << [addr, next_addr, fn]
end
addrs.sort!

calls = calls.map do |from, to|
  #_, _, fn = addrs.bsearch{|ca, na, fn|from < ca ? -1 : from >= na ? 1 : 0}
  fn = addrs.bsearch{|ca, na, fn|from < ca ? 1 : from >= na ? -1 : 0}
  fn = addrs[fn][2]
  [fn, to]
end

calls.each do |from, to|
  funcs[from][1] << to
  funcs[to][0] << from
end

$of = of
def show_call_tree(funcs, fn, stack, seen)
  $of.puts " " * stack.size + fn.to_s
  return if seen[fn]
  seen[fn] = true

  stack << fn
  funcs[fn][1].each do |callee|
    show_call_tree(funcs, callee, stack, seen)
  end
  stack.pop
end

funcs.each do |fn, kv|
  callers, callees = kv
  next if !callers.empty?

  show_call_tree(funcs, fn, [], {})
end
