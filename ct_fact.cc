template <int N>
struct compile_time_factorial {
  static const int result = N * compile_time_factorial<N-1>::result;
};

template <>
struct compile_time_factorial<0> {
  static const int result = 1;
};

#include <stdio.h>

int main() {
  const int v = compile_time_factorial<10>::result;
  printf("%d\n", v);
}
