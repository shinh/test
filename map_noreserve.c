#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
int main() {
  int64_t size = 128LL * 1024LL * 1024LL * 1024LL;
  printf("%p\n", malloc(size));
  printf("%p\n", mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  printf("%p\n", mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
}
