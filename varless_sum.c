#include <stdio.h>

int f() {
  if (2[(int*)__builtin_frame_address(0)])
    return (f(2[(int*)__builtin_frame_address(0)] - 1) +
            2[(int*)__builtin_frame_address(0)]);
  return 0;
}

int main() {
  printf("%d\n", f(100));
  return 0;
}
