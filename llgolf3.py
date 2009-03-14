from datetime import*
d,n=date.today(),0
while d.year<2014:
# if d.day==13 and d.weekday()==4:print d;n+=1
 if d.day==13 and d.weekday()==4:print d;n+=1
 d+=timedelta(1)
print n
