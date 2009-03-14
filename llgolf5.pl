s/^.?/$&+print/euntil/@ARGV/
#s/^.?/$&+print/efor(pop)x99

#s/^1?\d/$&-print/efor(pop)x99

#/.[^0]/||print$_,$/for 0..pop

#/^.0*$/&&print$_,$/for 0..pop



#s/^.0*$/$&
#/&&print for 0..pop



#$,=$/;print map/^.0*$/g,0..pop
#$\=$/;map{print/^.0*$/g}0..pop


#print grep/^.0*$/,($,=$/)..pop
#print grep{/^\d0*$/}0..pop,$,=$/


#print"$_
#"x/^.0*$/for 0..pop



#s|^.0*$|print$&.$/|efor 0..pop





#$\=$/;/^.0*$/||print for 0..pop

#map{print"$_\n"if/^.0*$/}0.."@ARGV";
#/^.0*$/&&print"$_\n"for 0.."@ARGV"
