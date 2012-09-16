a = ARGV[0].to_f
b = ARGV[1].to_f

aw = b - a * (b - 1)
bw = a - b * (a - 1)

puts "if A wins bookmaker takes %.3f%%" % (aw * 100 / (a + b))
puts "if B wins bookmaker takes %.3f%%" % (bw * 100 / (a + b))

ap = 100 / a
bp = 100 / b
puts "you can bet A if A wins >%.3f%%" % ap
puts "you can bet B if B wins >%.3f%%" % bp

ar = 1 / a
br = 1 / b
puts ("if A wins %.3f%%, A wins bo3 %.3f%%" %
      [ap, ((ar*ar + ar*(1-ar)*ar*2) * 100)])
puts ("if B wins %.3f%%, B wins bo3 %.3f%%" %
      [bp, ((br*br + br*(1-br)*br*2) * 100)])
