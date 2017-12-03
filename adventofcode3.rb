x = 0
y = 0
vx = 1
vy = 0
l = 1
c = 0
board = {}

nxt = proc{
  r = [x,y]

  v = 0
  [-1,0,1].each do |dy|
    [-1,0,1].each do |dx|
      nx = x + dx
      ny = y + dy
      if board[[nx,ny]]
        v += board[[nx,ny]]
      end
    end
  end
  if v == 0
    v = 1
  end

  board[r] = v

  x += vx
  y += vy
  c += 1
  if c == l
    c = 0
    if vx == 1 && vy == 0
      vx = 0
      vy = -1
    elsif vx == 0 && vy == -1
      vx = -1
      vy = 0
      l += 1
    elsif vx == -1 && vy == 0
      vx = 0
      vy = 1
    else
      vx = 1
      vy = 0
      l += 1
    end
  end

  #r
  v
}

1000.times{
  v = nxt[]
  if v > 265149
    p v
  end
}

__END__

x = 0
y = 0
vx = 1
vy = 0
l = 1
c = 0

nxt = proc{
  r = [x,y]
  x += vx
  y += vy
  c += 1
  if c == l
    c = 0
    if vx == 1 && vy == 0
      vx = 0
      vy = -1
    elsif vx == 0 && vy == -1
      vx = -1
      vy = 0
      l += 1
    elsif vx == -1 && vy == 0
      vx = 0
      vy = 1
    else
      vx = 1
      vy = 0
      l += 1
    end
  end

  r
}

a = nil
265149.times{
  a = nxt[]
}
p a[0].abs + a[1].abs

