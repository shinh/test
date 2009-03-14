class String
  def akr?
    # must not be capitalized
    return false unless self[/^[a-z]/]
    # there should be continuous upcases
    return false unless self[/[A-Z][A-Z]/]
    return true
  end
end

puts "defLATE: #{'defLATE'.akr?}"
puts "deflate: #{'deflate'.akr?}"
puts "Deflate: #{'Deflate'.akr?}"
puts "dEflate: #{'dEflate'.akr?}"
puts "DEFLATE: #{'DEFLATE'.akr?}"
puts "deFLate: #{'deFLate'.akr?}"

class String
  def akr
    akrize = proc{|s|s.split('').map{|_|rand<0.5 ? _.upcase : _.downcase} * ''}
    until (s = akrize[self]).akr?
    end
    s
  end
end

10.times{puts 'deflate'.akr}
