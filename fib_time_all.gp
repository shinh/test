set terminal gif
set output "fib_time_all.gif"
plot [0:230000] 'fib_time.dat' title "Haskell", 'fib_time_java.dat' title "Java", 'fib_time_rb.dat' title "Ruby 1.9", 'fib_time_cc.dat' title "C++"
