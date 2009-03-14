class Array
  def x
    self[0]
  end
  def y
    self[1]
  end
  def z
    self[2]
  end

  def rot_x
    [x, z, -y]
  end
  def rot_y
    [-z, y, x]
  end
  def rot_z
    [y, -x, z]
  end

  def rev_yz
    [-x, y, z]
  end
  def rev_zx
    [x, -y, z]
  end
  def rev_xy
    [x, y, -z]
  end

  def rot_yv(a)
    r = Math::PI * a / 180
    c = Math.cos(r)
    s = Math.sin(r)
    [c * x - s * z, y, c * z + s * x]
  end
  def rot_zv(a)
    r = Math::PI * a / 180
    c = Math.cos(r)
    s = Math.sin(r)
    [c * x + s * y, c * y - s * x, z]
  end

  def vec(p)
    [x-p.x, y-p.y, z-p.z]
  end

  def dot(p)
    [y*p.z-z*p.y, z*p.x-x*p.z, x*p.y-y*p.x]
  end
end

l0 = (Math.sqrt(5)+3) / 2
l1 = (Math.sqrt(5)+1) / 2
l2 = 1

# a = [[l0, -l2, 0], [l1, -l1, l1], [l2, 0, l0], [l1, l1, l1], [l0, l2, 0]]
# b = a.map(&:rev_xy).reverse
# c = a.map(&:rev_yz).reverse
# d = c.map(&:rev_xy).reverse
# e = a.map(&:rot_x).map(&:rot_y)
# f = e.map(&:rev_zx).reverse
# g = e.map(&:rev_xy).reverse
# h = g.map(&:rev_zx).reverse
# i = e.map(&:rot_y).map(&:rot_z)
# j = i.map(&:rev_yz).reverse
# k = i.map(&:rev_zx).reverse
# l = k.map(&:rev_yz).reverse

# dodeca = [a, b, c, d, e, f, g, h, i, j, k, l]

def rot(d)d=='x'?'y':d=='y'?'z':'x'end
a = [[l0, -l2, 0], [l1, -l1, l1], [l2, 0, l0], [l1, l1, l1], [l0, l2, 0]]
dodeca = [a,
          e=a.map(&:rot_x).map(&:rot_y),
          i=e.map(&:rot_y).map(&:rot_z)]
[[a, 'x'], [e, 'z'], [i, 'y']].each{|a, x|
  y = rot(x)
  z = rot(y)
  dodeca << b = a.map(&"rev_#{x}#{y}".to_sym).reverse
  dodeca << c = a.map(&"rev_#{y}#{z}".to_sym).reverse
  dodeca << d = c.map(&"rev_#{x}#{y}".to_sym).reverse
}

dodeca = dodeca.map{|pent|pent.map{|pt|pt.rot_zv(15).rot_yv(-15)}}

visible = dodeca.select{|pent|
  (pent[0].vec(pent[1])).dot(pent[2].vec(pent[1]))[0] >= 0
}
invisible = dodeca - visible

[[invisible, 0.8], [visible, 0]].each{|pents, color|
  puts "%f setgray" % color
  pents.each{|pent|
    pent.each_with_index{|pt, i|
      print pt[1,2].map{|v|(v*50)+300} * ' '
      puts i == 0 ? ' moveto' : ' lineto'
    }
    puts 'closepath'
    puts 'stroke'
  }
}
puts 'showpage'
