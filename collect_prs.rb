pr = nil

$<.each do |line|
  line.strip!
  next if line == ''
  if line =~ /Merge pull request (#\d+) from (\w+)/
    pr = [$1, $2]
  elsif pr
    puts "#{line} by @#{pr[1]} (#{pr[0]})"
    pr = nil
  end
end
