require 'socket'

srv = TCPServer.new(9999)
srv.listen(1)
sock = srv.accept

while true
  #sock.read(100)
  sleep(0.001)
end
