#!ruby
require 'irb/completion'
require 'irb/ext/save-history'
require 'tempfile'

IRB.conf[:SAVE_HISTORY] = 100000
IRB.conf[:HISTORY_FILE] = "#{ENV['HOME']}/.irb_history"

def disasm(b, arch = 'i386')
  tmp = Tempfile.new('irb_disasm', '/tmp')
  if b.class == Array
    tmp.print(b.pack("C*"))
  elsif b.class == String
    tmp.print(b)
  else
    raise "type mismatch (#{b.class})"
  end
  tmp.close
  if arch == 'z80'
    print `z80dasm -a -t #{tmp.path}`
  else
    print `objdump -m #{arch} -b binary -D #{tmp.path}`
  end
  nil
end
def asm(b, arch = 'i386')
  tmp = Tempfile.new('irb_asm', '/tmp')
  if arch == 'i8086'
    tmp.puts 'BITS 16'
  end
  tmp.print(b)
  tmp.close
  if arch == 'z80'
    system("z80asm #{tmp.path} -o /dev/stdout | od -t x1z -A x")
  elsif arch == 'i8086'
    system("nasm #{tmp.path} -o /tmp/irb_asm2")
    print `od -t x1z -A x /tmp/irb_asm2`
  else
    system("nasm #{tmp.path} -o /tmp/irb_asm2")
    print `od -t x1z -A x /tmp/irb_asm2`
  end
  nil
end
def xclip(s)
  pipe = IO.popen('xclip','w')
  pipe.print(s)
  pipe.close
  s
end

class Fixnum
  def combination(x)
    raise ArgumentError.new if self < x
    r = 1
    (x+1).upto(self){|i|r *= i}
    2.upto(x-1){|i|r /= i}
    r
  end

  def cuint
    self & ((1 << 32) - 1)
  end

  def cint
    v = cuint
    (v < (1 << 31)) ? v : v - (1 << 32)
  end

  def human
    if self >= 10_000_000_000
      "#{self/1000_000_000}G"
    elsif self >= 10_000_000
      "#{self/1000_000}M"
    elsif self >= 10_000
      "#{self/1000}k"
    else
      "#{self}"
    end
  end

  def fact
    n=1
    2.upto(self){|i|
      n*=i
    }
    n
  end
end
