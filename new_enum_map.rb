#!/usr/bin/env ruby

class Array
  alias :old_op_index :[]
  def [](a, *args)
    if a.class == Proc || a.respond_to?(:to_proc)
      map{|v|a.to_proc[v, *args]}
    else
      old_op_index(a, *args)
    end
  end
end

a = [*1..10]

p a.map{|v|v*v}
p a[->v{v*v}]

p a.map{|v|v.succ}
p a[:succ]

p a.map{|v|v*3}
p a[:*,3]
