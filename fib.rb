def fib_slow(n)
  if n<=1
    return 1
  else
    fib_slow(n-1)+fib_slow(n-2)
  end
end

#p fib_slow(46)

def fib(n)
  a=0
  b=1
  while n>0
    n-=1
    a,b=b,a+b
  end
  return a
end

def fib_fast(n)
  atot=1
  btot=0
  dtot=1
  a=1
  b=1
  d=0
  while n>0
    if n%2!=0
      atot,btot,dtot=atot*a+btot*b,atot*b+btot*d,btot*b+dtot*d
    end
    n/=2
    a,b,d=a*a+b*b,a*b+b*d,b*b+d*d
  end
  return btot
end

p fib_fast(1000000)
p fib(1000000)
