set terminal gif
set output "fib_time_rb.gif"
fit x*x*a 'fib_time_rb.dat' via a
b=2
FIT_LIMIT=1e-30
fit x**b*c 'fib_time_rb.dat' via b,c
plot 'fib_time_rb.dat', x*x*a, x**b*c
