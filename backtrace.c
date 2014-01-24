#include <stdio.h>
#include <string.h>
#include <execinfo.h>

void f() {
    void* buf[333];
    memset(buf, 0, 3*sizeof(void*));
    int cnt = backtrace(buf, 333);
    printf("cnt: %d\n", cnt);
    printf("builtin_return_address: %p %p\n",
           __builtin_return_address(0), __builtin_return_address(1));

    char** syms = backtrace_symbols(buf, cnt);

    int i;
    for (i = 0; i < cnt; i++) {
      printf("%p %s\n", buf[i], syms[i]);
    }
}

int main() {
    f();
}
