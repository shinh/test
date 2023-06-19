require 'json'

def github_word(lang, word)
  r = `gh api 'search/code?q=language:#{lang}+#{word}'`
  JSON.load(r)["total_count"]
end

def github_popularity(lang, keywords, filename)
  data = {}
  keywords.each do |word|
    count = github_word(lang, word)
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

java_keywords = %w(
abstract	continue	for	new	switch
assert	default	goto	package	synchronized
boolean	do	if	private	this
break	double	implements	protected	throw
byte	else	import	public	throws
case	enum	instanceof	return	transient
catch	extends	int	short	try
char	final	interface	static	void
class	finally	long	strictfp	volatile
const	float	native	super	while
)

#github_popularity("JavaScript", js_keywords - js_keywords_removed, "github_js_keywords.json")
github_popularity("Java", java_keywords, "github_java_keywords.json")
