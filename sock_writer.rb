require 'socket'

msg = 'h' * 1000

sock = TCPSocket.new('localhost', 9999)

sock.sync = false

i=0
while true
  p sock.write_nonblock('a'*100000000)
  p i
  i+=1
end
p i

sock.sync = true

i=0
while true
  i+=1
  puts "select"
  IO.select([], [sock])
  puts "write #{i}"
  sock.write(msg)
end
