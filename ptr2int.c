#include <stdint.h>
int main() {
  void* p = 0;
  int i = (int)p;
  long long j = (long long)p;
  uintptr_t k = (uintptr_t)p;
  unsigned long long l = k;
}
