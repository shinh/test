class String
  def -(s)
    gsub(s,'')
  end
end

puts "abc" - "b"
puts "abcabc" - "b"
puts "abcdef" - /b../
