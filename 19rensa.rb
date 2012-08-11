CHAIN_BONUS = [
  0, 0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384,
  416, 448, 480, 512,
]

LONG_BONUS = [
  0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7,
]

COLOR_BONUS = [
  0, 0, 3, 6, 12, 24,
]

def calc_score(chain)
  score = 0
  chain.size.times{|n|
    v = chain[n]
    num = v.inject(:+)
    chain_bonus = CHAIN_BONUS[n+1]
    color_bonus = COLOR_BONUS[v.size]
    long_bonus = v.inject(0){|r, a|
      r + LONG_BONUS[a]
    }
    bonus = chain_bonus + color_bonus + long_bonus
    if bonus == 0
      bonus = 1
    end
    score += 10 * num * bonus
    puts "#{n+1} #{score}"
  }
  score
end

chain19 = [[4]]*18 + [[6]]
p calc_score(chain19)

p calc_score([[4]]*14 + [[6,6,5,5]])

chain19 = [[4]]*18 + [[4]]
p calc_score(chain19)

p calc_score([[4]]*15 + [[4,4,4,4]])

