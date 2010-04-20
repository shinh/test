#include "inline_memfun.h"

#include <stdio.h>

void C::g(int n) {
    for (int i = 0; i < n; i++)
        printf("Hello\n");
    if (!n)
        printf("none\n");
}

void C::f(int n) {
    g(3);
    printf("foobar\n");
}

