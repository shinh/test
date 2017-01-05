# https://twitter.com/kagewaza_uswest/status/816593363952508928

def comb(n, k)
  a = (n-k+1..n).inject(1, &:*)
  b = (1..k).inject(1, &:*)
  a / b
end

NUM_TRIALS = 2140
EXPECTED = Rational(15, 1000)
NUM_SSRS_RANGE = (0 .. 21)

a = 0
NUM_TRIALS.times do |n|
  next if !NUM_SSRS_RANGE.include?(n)
  a += comb(NUM_TRIALS, n) * EXPECTED ** n * (1 - EXPECTED) ** (NUM_TRIALS - n)
end

p a.to_f
