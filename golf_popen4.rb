require 'fcntl'

if `arch` =~ /64/
  SYS_OPEN = 2
  SYS_CLOSE = 3
  SYS_DUP2 = 33
else
  SYS_OPEN = 5
  SYS_CLOSE = 6
  SYS_DUP2 = 63
end

def sys_open(fname, mode)
  fd = syscall(SYS_OPEN, fname, mode)
  if fd < 0
    raise
  end
  fd
end

def sys_close(fileno)
  if syscall(SYS_CLOSE, fileno) < 0
    raise
  end
  nil
end

def sys_dup2(old, new)
  rv = syscall(SYS_DUP2, old.fileno, new.fileno)
  if rv < 0
    raise "dup2"
  end
  return rv
end

def golf_popen4(*args)
  in_pipe = IO.pipe
  out_pipe = IO.pipe
  err_pipe = IO.pipe

  pid = fork do
    sys_dup2(in_pipe[0], STDIN)
    sys_dup2(out_pipe[1], STDOUT)
    sys_dup2(err_pipe[1], STDERR)

    (in_pipe + out_pipe + err_pipe).each do |f|
      f.close
    end

    sys_close 3 rescue nil
    sys_close 4 rescue nil

    sys_open("/dev/null", 0)
    sys_open("/dev/null", 2)

    exec(*args)
    raise
  end

  in_pipe[0].close
  out_pipe[1].close
  err_pipe[1].close
  return pid, in_pipe[1], out_pipe[0], err_pipe[0]
end

f = open("/dev/null")

#pid, stdin, stdout, stderr = Open4.popen4("ruby", "something.rb")
pid, stdin, stdout, stderr = golf_popen4("ruby", "something.rb")

p pid
p stdout.read
