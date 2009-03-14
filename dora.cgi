#!/usr/bin/env ruby

l = [
     'どこでも',
     '通り抜け',
     '空気',
     'スモール',
     'ビッグ',
     'もしも',
     'タイム',
    ]

puts %Q(Content-Type: text/html; charset=EUC-JP

<p>
#{l[rand(l.size)]}うんこ
</p>
<input type="submit" value="もう一度">
)
