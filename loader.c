#include <elf.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/auxv.h>

int main(int argc, char* argv[]) {
  int i;
  printf("argc=%d\n", argc);
  for (i = 0; i < argc; i++) {
    printf("argv[%d]=%s\n", i, argv[i]);
  }

  FILE* fp = fopen("/proc/self/maps", "rb");
  char buf[4096];
  fread(buf, 1, 4095, fp);
  puts(buf);
  fclose(fp);

  printf("AT_BASE=%lu\n", getauxval(AT_BASE));
  printf("AT_ENTRY=%lx\n", getauxval(AT_ENTRY));
  printf("AT_PHDR=%lx\n", getauxval(AT_PHDR));
  printf("AT_PHNUM=%lu\n", getauxval(AT_PHNUM));
  printf("AT_SECURE=%lu\n", getauxval(AT_SECURE));
}
