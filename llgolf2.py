#for i in re.findall(r'\b', 'hoge hoge'):sys.stdout.write(i)

#print''.join(map(capitalize,raw_input().split()))

s=raw_input();print''.join(map(min,s,s.title()))

#from sys import*
#s=read(0,99);print''.join(map(min,s,s.title()))
#s=raw_input()
#[output(s) for s in map(min,s,s.title())]


#import re
#print re.subn('\b.','\U',raw_input())[0]
#s=raw_input();print''.join(map(min,s,s.title()))
