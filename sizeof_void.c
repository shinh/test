#include <stdio.h>
int main() {
  void* p = NULL;
  p++;
  printf("%d %p\n", sizeof(void), p);
}

