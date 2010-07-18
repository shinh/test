set terminal gif
set output "fib_rss2.gif"
fit x*a 'fib_rss.dat' via a
plot 'fib_rss.dat', x*a
