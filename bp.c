#include <stdio.h>
int main() {
    void* p;
    asm("mov %%rbp, %0": "=g"(p));
    printf("%p %p\n", __builtin_frame_address(0), p);
}
