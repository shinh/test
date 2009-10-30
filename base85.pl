use Compress::Zlib;

# http://tools.ietf.org/html/rfc1924
sub decodeBase85($)
{
    my ($encoded) = @_;
    my %table;
    @characters = ('0'..'9', 'A'..'Z', 'a'..'z', '!', '#', '$', '%', '&', '(', ')', '*', '+', '-', ';', '<', '=', '>', '?', '@', '^', '_', '`', '{', '|', '}', '~');
    for (my $i = 0; $i < 85; $i++) {
        $table{$characters[$i]} = $i;
    }

    my $decoded = '';
    my @encodedChars = $encoded =~ /./g;

    #print @encodedChars+0, $/;

    for (my $encodedIter = 0; $encodedChars[$encodedIter];) {
        my $digit = 0;
        for (my $i = 0; $i < 5; $i++) {
            $digit *= 85;
            my $char = $encodedChars[$encodedIter];
            $digit += $table{$char};
            $encodedIter++;
        }

        for (my $i = 0; $i < 4; $i++) {
            $decoded .= chr(($digit >> (3 - $i) * 8) & 255);
        }
    }

    #return $decoded;
    return uncompress($decoded);
}


#print decodeBase85('cmcgoK@Nm42&C%Gi}VGyC*RS#Puc$;>j1^7i6@ht1lnN+#Fo7bq$tQ6Vr>y<!VQ4_
#n=M9AiCU{Da<TJsX5cb}GtdhmC+3(JcSKMWQsW^DWp)E(?RP^ICa{KJM6ym)p49Zt
#t*7gEj&Ea!K94a|!ZwboL2nB`_MvkjvlBIg|DPLNCVIRN3V^@Pf6oJ;{2S~5')

print decodeBase85('cmZ3>wTWv3Hw#N{etPOgM>$5;Q#<yZ=K=s#!w3!l')
#print decodeBase85('cmaDfkMYtx#tk1lCRcfiOy==S+g#=Of)xOJZVC1P')
