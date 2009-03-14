#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
void f() {
    puts("F");
}
int main() {
    //dlopen("a.out", RTLD_NOW);
    Dl_info info;
    if (dladdr(&f, &info)) {
        printf("%s\n", info.dli_sname);
    }
}
