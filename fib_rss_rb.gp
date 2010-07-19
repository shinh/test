set terminal gif
set output "fib_rss_rb.gif"
fit x*x*a 'fib_rss_rb.dat' via a
plot 'fib_rss_rb.dat', x*x*a
