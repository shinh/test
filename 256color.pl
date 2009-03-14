#!/usr/bin/perl

for (my $i = 0; $i < 256; $i++) {
    print pack('C*', '27'), '[38;5;', $i, 'm', $i, " ÆüËÜ¸ì\n";
}

for (my $i = 0; $i < 256; $i++) {
    print pack('C*', '27'), '[38;5;', $i, 'm', "#";
}
