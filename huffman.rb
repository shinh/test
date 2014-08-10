#!/usr/bin/env ruby

def huffman(stat)
  res = {}
  stat.each do |k, c|
    res[k] = ''
  end

  work = stat.map{|k, c|[[k], c]}
  while work[1]
    work = work.sort_by{|k, c|c}

    a = work.shift
    b = work.shift
    a[0].each do |k|
      res[k] = '0' + res[k]
    end
    b[0].each do |k|
      res[k] = '1' + res[k]
    end

    c = [a[0]+b[0], a[1]+b[1]]
    work.unshift(c)
  end

  stat.sort_by{|k, c|-c}.map{|k, c|[k, c, res[k]]}
end

def pretty_print_huffman(huff)
  orig_size = 0
  cmp_size = 0
  huff.each do |c, n, s|
    orig_size += 8 * n
    cmp_size += s.size * n
    p [c, n, s]
  end
  puts "#{orig_size / 8} => #{cmp_size / 8} (#{huff.size} kinds)"
end

if __FILE__ == $0
  text = $<.read
  stat = {}
  text.each_char{|c|
    stat[c] = 0 if !stat[c]
    stat[c] += 1
  }

  huff = huffman(stat)
  pretty_print_huffman(huff)
end
