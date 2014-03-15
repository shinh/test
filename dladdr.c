#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

void f() {
    puts("F");
}

int main() {
    Dl_info info;

    //dlopen("a.out", RTLD_NOW);
    if (dladdr(&f, &info) && info.dli_sname) {
        printf("%s\n", info.dli_sname);
    }
    printf("base of f: %p\n", info.dli_fbase);

    if (dladdr(&dladdr, &info)) {
        printf("base of dladdr: %p %s\n", info.dli_fbase, info.dli_fname);
    }
    if (dladdr(&printf, &info)) {
        printf("base of printf: %p %s\n", info.dli_fbase, info.dli_fname);
    }
    if (dladdr(&main, &info)) {
        printf("base of main: %p %s\n", info.dli_fbase, info.dli_fname);
    }
}
