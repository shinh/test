set log x
set terminal gif
plot "bijection_hash.dat" using 1:2 title "x*y" with lines, \
     "bijection_hash.dat" using 1:3 title "x+y*P" with lines, \
     "bijection_hash.dat" using 1:4 title "p(x,y)" with lines
