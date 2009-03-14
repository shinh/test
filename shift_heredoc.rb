# s='s'
# $><<s

undef:p 
def method_missing(s,*)$><<s
end
loop{eval(gets)<<$/}
