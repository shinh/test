set terminal gif
set output "fib_rss_all.gif"
plot [0:230000] 'fib_rss.dat' title "Haskell", 'fib_rss_java.dat' title "Java", 'fib_rss_rb.dat' title "Ruby 1.9", 'fib_rss_cc.dat' title "C++"
