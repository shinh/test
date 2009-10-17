#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
main(){printf("%p\n",dlsym(RTLD_DEFAULT,"puts"));}
