class B
  S=1
  @@s=1
  def B.s
    1
  end
end

class C
  S=2
  @@s=2
  def C.s
    2
  end
  class D < B
    def D.z
      p S
      p @@s
      p s
    end
  end
end

C::D.z
