#include <stdio.h>

#include "diamond.h"

A::A() { puts("A"); }
B::B() { puts("B"); }
C::C() { puts("C"); }
D::D() { puts("D"); }

int main() {
    D d;
    B b;

    printf("%p\n", (*(void***)&b)[-3]);
    printf("%p\n", (*(void***)&b)[-2]);
    printf("%p\n", (*(void***)&b)[-1]);
}
