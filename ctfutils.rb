#!/usr/bin/env ruby

require 'io/nonblock'

require 'socket'

if RUBY_VERSION !~ /^1\.8/
  Encoding.default_external = 'binary'
  Encoding.default_internal = 'binary'
end

def File.read(filename)
  File.open(filename, 'r:binary') do |f|
    f.read
  end
end

def File.write(filename, s)
  File.open(filename, 'w:binary') do |f|
    f.write(s)
  end
end

def install_io_log_hook(pipe, bn)
  log_in = "/tmp/#{bn}.in.log"
  log_out = "/tmp/#{bn}.out.log"
  File.delete(log_in) if File.exist?(log_in)
  File.delete(log_out) if File.exist?(log_out)
  orig_gets = pipe.method(:gets)
  orig_read = pipe.method(:read)
  orig_write = pipe.method(:write)
  pipe.define_singleton_method(:gets, proc do |*a|
                                 r = orig_gets[*a]
                                 File.open(log_in, 'a') do |of|
                                   of.write(r)
                                 end
                                 r
                               end)
  pipe.define_singleton_method(:read, proc do |*a|
                                 r = orig_read[*a]
                                 File.open(log_in, 'a') do |of|
                                   of.write(r)
                                 end
                                 r
                               end)
  pipe.define_singleton_method(:write, proc do |*a|
                                 orig_write[*a]
                                 File.open(log_out, 'a') do |of|
                                   of.write(*a)
                                 end
                               end)
end

def socket(*a)
  s = TCPSocket.new(*a)
  if RUBY_VERSION !~ /^1\.8/
    install_io_log_hook(s, a*':')
  end
  s
end

def popen(a)
  pipe = IO.popen(a, 'r+:binary')
  bn = File.basename(a)
  if RUBY_VERSION !~ /^1\.8/
    install_io_log_hook(pipe, bn)
  end
  pipe
end

class IO
  def get(n=nil)
    if n
      r = read(n)
      if !$quiet
        STDERR.print(r)
        #STDERR.puts ('%02x' * r.size) % r.unpack("c*")
      end
      r
    else
      l = gets
      STDERR.puts l if !$quiet
      l
    end
  end

  def puts(s)
    if s =~ /\n$/
      write(s)
    else
      write("#{s}\n")
    end
    flush
  end

  def p(*a)
    a.each do |s|
      puts s.inspect
    end
  end

  def show_all_buf
    while r = IO.select([self], [], [], 0)
      if r[0][0] == self
        STDERR.putc self.read(1)
      end
    end
  end

  def wait_until(reg)
    b = ''
    while true
      c = self.read(1)
      if !$quiet
        STDERR.putc c
      end
      b += c
      if reg =~ b
        return $~
      end
    end
  end

  def interactive
    STDOUT.puts 'INTERACTIVE!'
    begin
      while true
        r = IO.select([self, STDIN], [], [])
        if r[0][0] == self
          c = self.read(1)
          if c
            STDOUT.putc c
          else
            STDOUT.puts 'Connection closed (read)'
            return
          end
        else r[0][0] == STDIN
          input = STDIN.gets
          self.puts(input)
        end
      end
    rescue
      STDOUT.puts $!
      STDOUT.puts $!.backtrace
      STDOUT.puts ''
      STDOUT.puts 'output:'
      STDOUT.puts self.read
    end
  end
end

def proc_map(pid)
  if pid.is_a?(IO)
    IO.select([pid], [], [], 0.03)
    return proc_map(pid.pid)
  end

  File.read("/proc/#{pid}/maps")
end

class ProcMap
  def initialize(pid)
    @maps = []
    anon_id = 0
    proc_map(pid).each_line do |line|
      toks = line.split
      name = toks[5]
      if !name
        name = "*anonymous_#{anon_id}*"
        anon_id += 1
      end

      range = toks[0].split('-').map(&:hex)
      range = Range.new(range[0], range[1], true)
      prot = toks[1]
      @maps << [name, range, prot]
    end
  end

  def get_range(reg)
    first = nil
    last = nil
    @maps.each do |name, range, prot|
      if reg =~ name
        if !first || first > range.first
          first = range.first
        end
        if !last || last < range.last
          last = range.last
        end
      end
    end

    if !first || !last
      raise "Missing first or last: #{first} - #{last}"
    end

    Range.new(first, last, true)
  end
end

class Range
  def addr_str
    '%x-%x' % [first, last]
  end
end

class Lib
  attr_reader :syms

  def initialize(fn)
    @filename = fn
    @syms = {}
    IO.popen("nm -D #{@filename}").each do |line|
      toks = line.strip.split
      if toks.size == 3
        @syms[toks[2]] = toks[0].hex
      end
    end
  end

  def sym(n)
    @syms[n]
  end
end

class Binary
  attr_reader :entry, :is_pie, :filename, :is_cgc, :code, :maps, :is_exe

  def initialize(filename)
    @filename = filename
    @code = File.read(filename)
    if @code[0, 4] == "\x7fELF"
      parse_elf
    elsif @code[0, 4] == "\x7fCGC"
      #@code[0, 4] = "\x7fELF"
      #@code[7] = "\0"
      #@filename = "/tmp/#{File.basename(@filename)}"
      #File.write(@filename, @code)
      @is_cgc = true
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
    @is_exe = ehdr[0] != 1
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

def shellcode_from_dump(dump)
  dump = dump.gsub(/^\s*\h+:/, '')
  sc = ''
  dump.scan(/^\s*((\h{2}\s)+)/) do
    sc += $1.split.map{|h|h.hex.chr} * ''
  end

  if i = sc.index("\0")
    STDERR.puts "WARNING: NULL in shellcode at #{i}"
  end
  if i = sc.index("\n")
    STDERR.puts "WARNING: linebreak in shellcode at #{i}"
  end
  STDERR.puts "shellcode size: #{sc.size}"

  sc
end

def cookie(len)
  r = ''
  m = {}
  len.times{
    while true
      x = rand(26+26+10)
      x = if x < 10
            x.to_s
          elsif x < 10 + 26
            (?A.ord + x - 10).chr
          else
            (?a.ord + x - 10 - 26).chr
          end
      if r.size > 3
        l = r[-3..-1] + x
        if m[l]
          #STDERR.puts 'retrying'
          next
        end
        m[l] = true
      end
      r += x
      break
    end
  }
  r
end

if ARGV[0] == '-p'
  $prod = true
  ARGV.shift
end
if ARGV[0] == '-g'
  $gdb = true
  ARGV.shift
end

# from http://rosettacode.org/wiki/Modular_inverse#Ruby

def extended_gcd(a, b)
  last_remainder, remainder = a.abs, b.abs
  x, last_x, y, last_y = 0, 1, 1, 0
  while remainder != 0
    last_remainder, (quotient, remainder) = remainder, last_remainder.divmod(remainder)
    x, last_x = last_x - quotient*x, x
    y, last_y = last_y - quotient*y, y
  end

  return last_remainder, last_x * (a < 0 ? -1 : 1)
end

def invmod(e, et)
  g, x = extended_gcd(e, et)
  if g != 1
    raise 'Teh maths are broken!'
  end
  x % et
end

def gnu_hash(n)
  h = 5381
  n.each_byte{|c|
    h = (h * 33 + c) & ((1 << 32) - 1)
  }
  h
end
