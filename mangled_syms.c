// Example output:
//
// libc text: 0x7f40049a7000 0x7f4004b58000 /lib/x86_64-linux-gnu/libc-2.27.so
// libc data: 0x7f4004d5b000 0x7f4004d5d000 /lib/x86_64-linux-gnu/libc-2.27.so
// 0x7f40049de910
// exit /lib/x86_64-linux-gnu/libc.so.6

#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>

void** demangle(void* p, unsigned long rnd) {
  unsigned long v = (unsigned long)p;
  v = ((v >> 17UL) | (v << 47UL)) ^ rnd;
  return (void**)v;
}

int main() {
  unsigned long* at_random = (unsigned long*)getauxval(AT_RANDOM);
  unsigned long rnd = at_random[1];

  atexit((void(*)(void))exit);

  char buf[999];

  void** libc_begin;
  void** libc_end;
  FILE* fp = fopen("/proc/self/maps", "rb");
  while (fgets(buf, 998, fp)) {
    void** begin;
    void** end;
    {
      char r, w, x, p;
      long offset, inode;
      int major, minor;
      char name[999];
      name[0] = 0;
      sscanf(buf, "%p-%p %c%c%c%c %lx %d:%d %ld%s",
             &begin, &end, &r, &w, &x, &p, &offset, &major, &minor, &inode,
             name);

      if (x == 'x' && strstr(name, "/libc-")) {
        fprintf(stderr, "libc text: %p %p %s\n", begin, end, name);
        libc_begin = begin;
        libc_end = end;
      }
      if (w == 'w' && strstr(name, "/libc-")) {
        fprintf(stderr, "libc data: %p %p %s\n", begin, end, name);
        for (void** p = begin; p < end; p++) {
          void** demangled = demangle(*p, rnd);
          if (libc_begin <= demangled && demangled < libc_end) {
            fprintf(stderr, "%p\n", demangled);
            Dl_info info;
            if (dladdr(demangled, &info) != 0) {
              fprintf(stderr, "%s %s\n", info.dli_sname, info.dli_fname);
            }
          }
        }
      }
    }
  }
  fclose(fp);
}
