#include <stdio.h>

void other(void (*funcp)()){
  char* p = (char*)funcp;
  int i;
  for (i = 0; i < 10; i++) {
      printf("%c", p[i]);
  }
  //funcp();
}

void outer(void){
  asm("#");
  int a = 12;
  void inner(void){ printf("other a is %d\n", a); }
  asm("#");
  other(inner);
  asm("#");
}

int main() {
    outer();
}
