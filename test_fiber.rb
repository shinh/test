def f(i,j)
  Fiber.yield
  i < 30000 ? f(i+j,j) : 0
end

f=Fiber.new{p f(0,1)}
#p f(0,1)

while true
  p 'hoge'
  f.resume
end
