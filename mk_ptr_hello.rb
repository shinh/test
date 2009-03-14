s="`j X$@P[PYPPPPX4.4 PZUX, P^XH,=)F(P_X3F()8)8@)8@@)8)8@PYX@@@@CQBaGHello, world!\n"

i=0
while v=s[i,4]
  break if !v[0]
  r = '0x%x' % v.unpack('i')[0]
  if i == 0
    puts "char* main = (char*)#{r}U;"
  else
    puts "char* main#{i/4} = (char*)#{r}U;"
  end
  i+=4
end
