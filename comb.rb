#a=10**9
#b=2*10**9
a=15634
b=15634+456
v=1
1.upto(a) do |i|
  v*=b-i+1
end
1.upto(a) do |i|
  v/=i
end

p v%4990
