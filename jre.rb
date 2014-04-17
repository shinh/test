# -*- coding: utf-8 -*-
# Japanese Rspec East

class Ha
  def initialize(a, b)
    @a = a
    @b = b
    if @a != @b
      @e = "FAIL"
    end
  end
  def でない
    @e = @a == @b && "FAIL"
    self
  end
  def より小さい
    @e = @a >= @b && "FAIL"
    self
  end
  def より大きい
    @e = @a <= @b && "FAIL"
    self
  end
  def 以下
    @e = @a > @b && "FAIL"
    self
  end
  def 以上
    @e = @a < @b && "FAIL"
    self
  end
  def です
    puts @e || 'PASS'
  end
end

class Object
  def は(x)
    Ha.new(self, x)
  end
end

3.は(2 + 1).です
3.は(2 + 2).でない.です
3.は(2 + 2).より小さい.です
3.は(1 + 1).以上.です
