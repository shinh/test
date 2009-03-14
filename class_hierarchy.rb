h = {}
ObjectSpace.each_object(Class){|c| (h[c.superclass] ||= []) << c}
table = h
table[nil]      # => [Object]
table[Integer]  # => [Bignum, Fixnum]
table[NilClass] # => nil

def class_hierarchy(table, indent=0, sup=nil)
  excepts = [Exception]
  return unless table[sup]
  table[sup].sort_by{|c| c.name}.each do |c|
    puts "#{'  '*indent}#{c}"
    class_hierarchy(table, indent+1, c) unless excepts.include? c
  end
end
class_hierarchy(table)
