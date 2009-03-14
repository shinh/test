"Hello, world!\n".each_byte{|v|
  puts "(''<<"+"-~"*(v)+"_)+"
}
