#!/usr/bin/env ruby

$: << File.dirname(__FILE__)

require_relative 'ctfutils'

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

exe = Binary.new(file)

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
  dumpopt = '-S'
  if !exe.is_exe
    dumpopt += 'r'
  end
  if exe.is_cgc
    cmd = "i386-linux-cgc-objdump #{dumpopt} #{exe.filename}"
  elsif `file #{file}` =~ /PE32/
    cmd = "ruby #{File.dirname(File.realpath(__FILE__))}/dispe.rb #{exe.filename}"
  elsif `file #{file}` =~ / ARM/
    cmd = "arm-linux-gnueabihf-objdump #{dumpopt} #{exe.filename}"
  elsif `file #{file}` =~ / SH,/
    cmd = "/usr/local/stow/binutils-all/bin/all-objdump #{dumpopt} #{exe.filename}"
  else
    cmd = "objdump #{dumpopt} #{exe.filename}"
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

func_prolog = /push\s+(%[er]bp|{lr})/
dump.each_line do |line|
  if line =~ /^(\h+) <(.*)>:$/
    labels[$1.hex] = $2
  end
  if line =~ /endbr64/
    func_prolog = /endbr64/
  end
end

dump.each_line do |line|
  if line =~ /^(\h+) <(.*)>:$/
    labels[$1.hex] = $2
  elsif line =~ func_prolog
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
    is_func = label =~ /func/
    if !is_func || exe.is_exe
      if is_func
        of.puts ""
      end
      of.puts "#{label}:"
    end
  end

  if line =~ /^\s*(\h+):/ && c = comments[[addr, false]]
    of.puts c.map{|l|
      "# #{l}\n"
    } * ""
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
    if annot[0] !~ /func/ || exe.is_exe
      line += " ; #{annot * ' '}"
    end
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
  fn = addrs.bsearch{|ca, na, fn|from < na}
  [fn[2], to]
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
