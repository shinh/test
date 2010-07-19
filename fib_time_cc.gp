set terminal gif
set output "fib_time_cc.gif"
fit x*x*a 'fib_time_cc.dat' via a
b=2
FIT_LIMIT=1e-50
fit x**b*c 'fib_time_cc.dat' via b,c
plot 'fib_time_cc.dat', x*x*a, x**b*c
