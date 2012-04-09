#include <stdarg.h>

void f(int i, ...) {
  va_list ap;
  va_start(ap, i);
  for (i = 0; i < 10; i++) {
    double v = va_arg(ap, double

);
  }
  va_end(ap);
}

int main() {
  puts("hello-");
}
