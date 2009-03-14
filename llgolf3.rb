#!/usr/bin/env ruby

#require'date';i=0;Date.today.upto(Date.new(2014)-1){|d|puts d.strftime'%Y-%m-%d'if d.mday*d.wday==65&&i+=1};puts i


#Time.mktime(2004)
#Time.at(1388502000)
#Time.at(0x52c2dbf0)


#p (Time.now..Time.gm(2013,12,31)).step(86400).count{|x|x.mday*x.wday==65&&!puts(x.strftime'%Y-%m-%d')}

#c=0
(Time.now...Time.gm(14)).step(86400){|d|"#{d}"=~/i\D*13/&&p(d)|$.+=1}
p$.

#t=Time.now;t.day*t.wday==65&&(puts t.strftime'%Y-%m-%d';$.+=1)while Time.mktime(2014)>t+=86400;p$.-1

#require'date';i=0;Date.today.upto(Date.new(2014)-1){|d|puts d.strftime'%Y-%m-%d'if d.mday*d.wday==65&&i+=1};puts i


# require 'date'

# from = DateTime.now
# to = DateTime.parse("2013-12-31")

# friday = (from..to).inject(0) do |friday, date|
#     if date.mday == 13 and date.wday == 5 then
#         puts date.strftime('%Y-%m-%d')
#         friday + 1
#     else
#         friday
#     end
# end

# p friday
