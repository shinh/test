BEGIN {
  print "o, ";
  END {
    print "wor";
    BEGIN {
      print "ll";
      END {
        print "ld";
        BEGIN {
          print "He";
          END {
            print "!\n";
          }
        }
      }
    }
  }
}
