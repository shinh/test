# class Fixnum
#   def times
#     i = 0
#     s = self
#     while i < s
#       yield i
#       i += 1
#     end
#   end
# end

s = 0
1000000.times{|i|
  s+=i
}
p s
