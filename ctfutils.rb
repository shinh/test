#!/usr/bin/env ruby

require 'io/nonblock'

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

def popen(a)
  IO.popen(a, 'r+:binary')
end

class IO
  def get(n=nil)
    if n
      r = read(n)
      STDERR.puts ('%02x' * r.size) % r.unpack("c*") if !$quiet
      r
    else
      l = gets
      STDERR.puts l if !$quiet
      l
    end
  end

  def show_all_buf
    while r = IO.select([self], [], [], 0)
      if r[0][0] == self
        STDOUT.putc self.read(1)
      end
    end
  end

  def interactive
    STDOUT.puts 'INTERACTIVE!'
    begin
      while true
        r = IO.select([self, STDIN], [], [])
        if r[0][0] == self
          STDOUT.putc self.read(1)
        else r[0][0] == STDIN
          input = STDIN.gets
          self.puts(input)
        end
      end
    rescue
      STDOUT.puts $!
      STDOUT.puts $!.backtrace
      STDOUT.puts
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
