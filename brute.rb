# ! !!?!
#oaRy2a0

l=[*'0'..'9']+[*'a'..'z']+[*'A'..'Z']
l.each do |a|
  l.each do |b|
#    l.each do |c|
      k = "#{a}a#{b}y2a0"
      system("echo #{k}\r | wine FSC08_Level1.exe > out/#{k}&")
#    end
  end
end

