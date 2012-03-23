#include <stdio.h>
main(){
  int i = 0;
  printf("Hello,");
  for (i = 0; i < 68000; i++) {
    fprintf(stderr, "%c", 'a');
  }
  puts(" world!");
}
