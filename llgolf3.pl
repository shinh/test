use Date::Simple':all';
for($t=today;$t++!~2014;)
{$t->day_of_week*$t->day-65||print"$t\n"}


#use Date::Simple':all';
#for($t=today;$t++<d8"20131231";)
#{print"$t\n"if$t->day_of_week*$t->day==65}


#use Date::Simple':all';
#for($t=today;$t++<d8"20131231";)
#{$t->day_of_week*$t->day-65||print"$t\n"}
