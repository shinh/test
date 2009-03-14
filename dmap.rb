module Enumerable
  def dmap
    DelegateMap.new(self)
  end
end

class DelegateMap < BasicObject
  def initialize(enum)
    @enum = enum
  end
  def method_missing(mhd, *args, &blk)
    @enum.map {|elem| elem.__send__(mhd, *args, &blk) }
  end
end

def f(x)
  x+1
end

def g(x)
  y=x+1
  y+1
end

p [1].dmap+1
p f([1].dmap)
#p g([1].dmap)

#p [[2]].dmap.dmap+1
p [[2]].dmap.dmap.dmap+1
