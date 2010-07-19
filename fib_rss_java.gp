set terminal gif
set output "fib_rss_java.gif"
fit x*x*a 'fib_rss_java.dat' via a
plot 'fib_rss_java.dat', x*x*a
