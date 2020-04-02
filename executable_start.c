#include <stdio.h>

#if defined(__APPLE__)
#if 1
extern char _mh_execute_header[1];
#define __executable_start (_mh_execute_header - 0x100000000)
#else
char* __executable_start = nullptr;
#endif
#else
extern char __executable_start[1] __attribute__((weak));
#endif

int main() {
    printf("%zx %p\n", (char*)main - __executable_start, __executable_start);
}