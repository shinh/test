#include <stdio.h>
#include <string.h>
#include <execinfo.h>

void f() {
    void* buf[333];
    memset(buf, 0, 3*sizeof(void*));
    printf("%d\n", backtrace(buf, 333));
    printf("%p %p %p\n", buf[0], buf[1], buf[2]);
    printf("%p %p\n", __builtin_return_address(0), __builtin_return_address(1));

    char** syms = backtrace_symbols(buf, 333);
    printf("%s %s %s\n", syms[0], syms[1], syms[2]);
}

int main() {
    f();
}
