Signal.trap("CLD")  { puts "Child died" }
fork {sleep 1} && Process.wait
sleep 1
