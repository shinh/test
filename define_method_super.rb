class A
  def hoge
    "hoge"
  end
end

class B < A
  define_method :hoge do
    super()
  end
end

p B.new.hoge
