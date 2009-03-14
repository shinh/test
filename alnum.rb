class Object
  def f a
    concat 100
  end
end
f String nil

# class String
#   alias originalinspect inspect
#   def inspect
#     concat 100
#   end
#   alias inspect originalinspect
# end

# p String nil
