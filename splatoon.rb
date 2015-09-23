# -*- coding: utf-8 -*-
WEAPON_MAP = {
  'カーボン' => 'カーボンローラー',
  'シューコラ' => 'シューターコラボ',
  'スプラシューターコラボ' => 'シューターコラボ',
}

while true
  case ARGV[0]
  when '-l'
    ARGV.shift
    $len = ARGV[0].to_i
    ARGV.shift
  else
    break
  end
end

def parse_results(s)
  a = []
  s.scan(/^(\d+) (\d+) ([wl])/) do
    a << [$1, $2, $3]
  end
  a
end

def per(v, t)
  '%.2f%%' % (v * 100.0 / t)
end

def calc_stat(a)
  if $len && a.size > $len
    a = a[-$len..-1]
  end

  num = 0
  win = 0
  lose = 0
  kill = 0
  death = 0
  a.each do |k, d, wl|
    num += 1
    if wl == 'w'
      win += 1
    else
      lose += 1
    end
    kill += k.to_i
    death += d.to_i
  end
  [num, win, lose, kill, death]
end

def show_stat(title, a)
  num, win, lose, kill, death = *calc_stat(a)
  puts "#{title}: #{num} matches"
  puts "win #{per(win,num)}"
  puts "k/d %.2f" % (kill.to_f/death)
  puts "k/m %.2f" % (kill.to_f/a.size)
  puts "d/m %.2f" % (death.to_f/a.size)
  puts
end

def filter(criteria, all_results)
  a = []
  all_results.each do |tags, results|
    if criteria.all?{|t|tags.include?(t)}
      a += results
    end
  end
  a
end

def show_stat_of(criteria, all_results)
  a = filter(criteria, all_results)
  if a.empty?
    return
  end

  title = criteria * ' '
  title = 'TOTAL' if title.empty?
  show_stat(title, a)
end

all_results = []
all_tags = {}
all_weapons = {}

memo = File.read(ARGV[0] || "#{ENV['HOME']}/memo/splatoon")
memo.split("\n\n").each do |s|
  lines = s.split("\n")
  title = lines[0]
  tags = title.split.map do |t|
    if WEAPON_MAP[t]
      WEAPON_MAP[t]
    else
      t
    end
  end
  weapon = tags[0]
  all_weapons[weapon] = true
  cont = lines[1..-1] * "\n" + "\n"

  tags << (title =~ /@/ ? 'NAWA' : 'GATI')
  tags.each{|t|all_tags[t] = true}
  all_results << [tags, parse_results(cont)]
end

all_tags.reject!{|t|t =~ /^[CBAS][-+]?(\d+|XX)/}

all_tags.sort.each do |t, _|
  show_stat_of([t], all_results)
end

all_weapons.each do |w, _|
  'AYH'.each_char do |c|
    show_stat_of(['GATI', w, c], all_results)
  end
end
show_stat_of(['NAWA'], all_results)
show_stat_of(['GATI'], all_results)
show_stat_of([], all_results)

kdmap = {}
max_kill = 15
max_death = 15
filter(['GATI'], all_results).each do |results|
  k, d, wl = *results
  if k !~ /\d/ || d !~ /\d/
    raise "#{k} #{d} #{wl}"
  end

  k = k.to_i
  d = d.to_i
  if k > max_kill
    k = max_kill
  end
  if d > max_death
    d = max_death
  end
  kdmap[[k, d]] = [0, 0] if !kdmap[[k, d]]
  if wl == 'w'
    kdmap[[k, d]][0] += 1
  elsif wl == 'l'
    kdmap[[k, d]][1] += 1
  end
end

File.open('kdmap.html', 'w') do |of|
  num, win, lose, kill, death = *calc_stat(filter(['GATI'], all_results))
  of.puts "<p>ガチソロ#{num}試合 勝率#{per(win,win+lose)} k/d=#{"%.2f"%(kill.to_f/death)}"

  of.puts '<p><table border=1>'
  of.puts '<tr>'
  of.puts '<th>kill \ death'
  0.upto(max_death) do |d|
    if d == max_death
      d = "#{d}+"
    end
    of.puts "<th>#{d}"
  end
  of.puts '</tr>'

  0.upto(max_kill) do |k|
    of.puts '<tr>'
    of.puts %Q(<th>#{k == max_kill ? "#{k}+" : k})
    0.upto(max_death) do |d|
      w, l = *kdmap[[k, d]]
      if w
        r = w.to_f / (w+l)
        c = '#%02x%02xa0' % [(255 - 90 * r).to_i, (255 - 90 * (1.0 - r)).to_i]
        of.puts %Q(<td style="background-color: #{c}">#{w}/#{w+l} #{per(w,w+l)}</td>)
      else
        of.puts "<td>N/A"
      end
    end
  end

  of.puts '</table>'
end
