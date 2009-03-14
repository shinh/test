puts <<END
class String
  alias originalinspect inspect
  def inspect
    alias inspect originalinspect
    undef originalinspect
    #{s='';$<.read.each_byte{|v|s+="concat #{v}\n"};s}
    raise self
  end
end
begin
  p String nil
rescue => c
  eval String c
end
END
