moji = []
10.times{|i|moji.push(i.to_s)}
26.times{|i|moji.push((?A+i).chr)}

S = 100

File.open('moji.txt') do |f|
  (10+26).times{|m|
    s = moji[m]
    File.open('csv/%s.csv' % s, 'w') do |of|
      of.puts '"%s","X","Y"' % s
      4.downto(0){|y|
        l = f.gets
        4.downto(0){|x|
          if l[x] == ?.
            of.puts "1,#{x*S},#{y*S}"
            of.puts "2,#{x*S+S},#{y*S}"
            of.puts "2,#{x*S+S},#{y*S+S}"
            of.puts "2,#{x*S},#{y*S+S}"
          end
        }
      }
    end
  }
end
