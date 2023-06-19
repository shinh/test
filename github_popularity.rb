require 'json'

def github_word(word)
  r = `gh api 'search/code?q=language:JavaScript+#{word}'`
  JSON.load(r)["total_count"]
end

def github_popularity(keywords, filename)
  data = {}
  keywords.each do |word|
    count = github_word(word)
    puts "#{word} #{count}"
    data[word] = count
    sleep 10
  end

  File.open(filename, "w") do |of|
    of.print JSON.dump(data)
  end
end

js_keywords = %w(abstract	arguments	await	boolean
break	byte	case	catch
char	class	const	continue
debugger	default	delete	do
double	else	enum	eval
export	extends	false	final
finally	float	for	function
goto	if	implements	import
in	instanceof	int	interface
let	long	native	new
null	package	private	protected
public	return	short	static
super	switch	synchronized	this
throw	throws	transient	true
try	typeof	var	void
volatile	while	with	yield)

js_keywords_removed = %w(abstract	boolean	byte	char
double	final	float	goto
int	long	native	short
synchronized	throws	transient	volatile)

github_popularity(js_keywords - js_keywords_removed, "github_js_keywords.json")
