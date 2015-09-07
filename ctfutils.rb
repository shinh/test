#!/usr/bin/env ruby

require 'io/nonblock'

require 'socket'

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
  install_io_log_hook(s, a*':')
  s
end

def popen(a)
  pipe = IO.popen(a, 'r+:binary')
  bn = File.basename(a)
  install_io_log_hook(pipe, bn)
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
