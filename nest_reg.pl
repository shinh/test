#!perl -l
$r = qr{[a-z]|\((??{$r})\)}x;
print($r);
print('(((a)+b)'=~m%$r([-+*/])$r%);
print("$1 @-");
print("@+");
