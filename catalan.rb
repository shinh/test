def catalan(n)
  if n == 0
    return 1
  end
  return ((4*n - 2)*catalan(n-1))/(n+1)
end
def countDiagrams(n)
  return catalan(n+1)-1
end
p countDiagrams(30)
