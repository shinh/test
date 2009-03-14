#include <stdio.h>
int main(char a) {
  char* p = &a;
  int b = 1 << 30;
  if (p > p + b) {
    printf("GCC 4.2 or lower\n");
  }
  else {
    printf("GCC 4.3\n");
  }
}
