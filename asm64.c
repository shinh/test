#include <stdio.h>
void f() {
    void* ebp;
    asm("mov %%rbp, %0": "=m"(ebp));
/*
    asm("mov %%rbp, %%rax;\n"
        "mov %%rax, (%0);\n"
        "mov %%rax, 4(%0);\n"
        : "=m"(&ebp));
*/
    printf("%p\n", ebp);
    printf("%p\n", __builtin_frame_address(0));
    printf("%p\n", ((void**)ebp)[0]);
    printf("%p\n", __builtin_frame_address(1));
}
int main() {
    f();
}
