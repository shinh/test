require 'json'

pop = JSON.load(File.read(ARGV[0]))

if ARGV[0] == "github_js_keywords.json"
  js_keywords_removed = %w(abstract	boolean	byte	char
double	final	float	goto
int	long	native	short
synchronized	throws	transient	volatile)
  js_keywords_removed.each do |word|
    pop.delete(word)
  end
end

tot = 0
pop.each do |k, v|
  tot += v
end

t10 = 0
i = 0
pop.sort_by{|k, v|-v}[0,50].each do |k, v|
  puts "#{i+=1}\t#{k}\t#{v}"
  t10 += v
end
puts "other\t#{tot - t10}"

