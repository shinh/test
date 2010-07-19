set terminal gif
set output "fib_rss_cc.gif"
fit x*x*a 'fib_rss_cc.dat' via a
plot 'fib_rss_cc.dat', x*x*a
