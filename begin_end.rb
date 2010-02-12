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

__END__

END {
  print "wor";
  BEGIN {
    print "o, ";
    END {
      print "ld";
      BEGIN {
        print "ll";
        END {
          print "!\n";
          BEGIN {
            print "He";
          }
        }
      }
    }
  }
}
