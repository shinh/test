def med(a,b,c,d,e)
  [[a,b,c].max,
   [a,b,d].max,
   [a,b,e].max,
   [a,c,d].max,
   [a,c,e].max,
   [a,d,e].max,
   [b,c,d].max,
   [b,c,e].max,
   [b,d,e].max,
   [c,d,e].max].min
end

a=[1,2,3,4,5]

100000.times do
  a.shuffle!
  if med(*a) != 3
    raise a
  end
end
