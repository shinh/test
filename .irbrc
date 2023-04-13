#!ruby

require 'irb/completion'
require 'irb/ext/save-history'
require 'irb/inspector'

autoload :JSON, 'json'
autoload :Tempfile, 'tempfile'

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
alias disas disasm

def asm(b, arch = 'i386')
  tmp = Tempfile.new('irb_asm', '/tmp')
  if arch == 'i8086'
    tmp.puts 'BITS 16'
  elsif arch == 'i386'
    tmp.puts 'BITS 32'
  elsif arch == 'x86_64'
    tmp.puts 'BITS 64'
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

def combination(x, y)
  raise ArgumentError.new if x < y || y < 0
  r = 1
  (x - y + 1).upto(x){|i|r *= i}
  2.upto(y){|i|r /= i}
  r
end

class Integer
  def cuint
    self & ((1 << 32) - 1)
  end

  def cint
    v = cuint
    (v < (1 << 31)) ? v : v - (1 << 32)
  end

  def human
    if self >= 10_000_000_000_000_000
      "#{(self/1000_000_000_000_000).to_i}P"
    elsif self >= 10_000_000_000_000
      "#{(self/1000_000_000_000).to_i}T"
    elsif self >= 10_000_000_000
      "#{(self/1000_000_000).to_i}G"
    elsif self >= 10_000_000
      "#{(self/1000_000).to_i}M"
    elsif self >= 10_000
      "#{(self/1000).to_i}k"
    else
      "#{self}"
    end
  end

  def to_x
    '%x' % self
  end

  def to_dg
    r = ''
    v = self
    if v < 0
      r = '-'
      v = -v
    end
    while v >= 1000
      r = '_%03d' % (v % 1000) + r
      v /= 1000
    end
    r = v.to_s + r
    r
  end

  def fact
    n=1
    2.upto(self){|i|
      n*=i
    }
    n
  end
end

def cookie(n)
  r = 0
  n.times{|i|r += 1.15 ** i}
  r
end

def Time.to_us(*a)
  if a.size == 3 && a[0] < 24
    a = [2000,1,1] + a
  end
  t = Time.local(*a)

  tz = ENV['TZ']
  ENV['TZ'] = 'US/Pacific'
  t = Time.at(t.to_i)
  t.inspect  # let the timezone fixed
  ENV['TZ'] = tz
  t
end

def bin
  Encoding.default_external = 'binary'
  Encoding.default_internal = 'binary'
end

def get_irb_inspectors
  if IRB.constants.include?(:INSPECTORS)
    IRB::INSPECTORS
  else
    IRB::Inspector::INSPECTORS
  end
end

def float_bits(v)
  s = [v].pack('d').bytes
  exponent = ((s[7] & 0x7f) << 4) | ((s[6] & 0xf0) >> 4)
  mantissa = s[6] & 0xf
  5.downto(0) do |i|
    mantissa <<= 8
    mantissa |= s[i]
  end
  [exponent, mantissa]
end

def prettify(v)
  fyi = []
  if v.is_a?(Rational)
    fyi << v.to_f
    fyi << v
    v = v.to_i
  end
  if v.is_a?(Integer) && v <= 2**257
    fyi << '0x%x' % v
    if v > 0 && v < 127
      fyi << "%s" % v.chr.inspect
    end
    if v > 1000 && v < 2 ** 65
      fyi << v.human
    end
    "%s (%s)" % [v, fyi * ' ']
  elsif v.is_a?(Float)
    "%s (%s)" % [v, float_bits(v) * ' ']
  elsif v.is_a?(String) && v.size == 1
    "%s (%d)" % [v, v.ord]
  else
    v
  end
end

inspector_proc = proc{|v|
  pv = prettify(v)
  # if pv != v
  #   pv
  # else
  #   get_irb_inspectors[true].inspect_value(pv)
  # end
}
get_irb_inspectors['mine'] = IRB::Inspector(inspector_proc)
IRB.conf[:INSPECT_MODE] = 'mine'

def frenzy_lucky(cps, bank = 0)
  [cps * 900 * 7, bank * 0.15, cps * 900 * 7 / 0.15]
end

def l(a)
  system("lv -ci #{a}")
end

def maseki(a)
  a = a.sort_by{|v|-v}
  s = 0
  a.each_with_index do |v, i|
    s += (v.to_f / 2 ** i).ceil
  end
  s
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

def modpow(a, b, n)
  r = 1
  while b > 0
    if (b & 1) == 1
      r = (r * a) % n
    end
    b >>= 1
    a = (a * a) % n
  end
  r
end
