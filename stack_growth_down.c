#include <stdio.h>

static bool StackGrowsDown(int *x) {
  int y;
  return &y < x;
}

int main() {
    int x;
    printf("%d\n", StackGrowsDown(&x));
}
