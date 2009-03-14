def make_pure_enumrator(e)
  def e.inspect
    "#<Enumerator::#{to_a.inspect}>"
  end
  e
end

class Array
  alias orig_each each
  def each
    make_pure_enumrator(orig_each)
  end
end

class Range
  alias orig_each each
  def each
    make_pure_enumrator(orig_each)
  end
end

p [1,2,3].each
p (0..3).each
