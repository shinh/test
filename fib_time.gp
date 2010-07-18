set terminal gif
set output "fib_time2.gif"
fit x*x*a 'fib_time.dat' via a
b=2
fit x**b*c 'fib_time.dat' via b,c
plot 'fib_time.dat', x*x*a, x**b*c
