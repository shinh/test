#include <stdio.h>

#include <stdarg.h>
#include <stdint.h>

void showstack() {
    void* v;
    void** start = &v;
    void** p = start;
    void** end = (void**)(((intptr_t)p + 4095) & ~4095);
    for (; p <= end; p++) {
        printf("%p: %p %d %f", p, *p, *(int*)p, *(double*)p);
        void** pp = (void**)*p;
        if (p < pp && pp <= end) {
            printf(" stack");
        }
        puts("");
    }
}

void f(int i,...) {
/*
    va_list ap;
    asm("# va_start");
    va_start(ap, i);
    asm("# va_arg");
    int k = va_arg(ap, int);
    asm("# va_end");
    va_end(ap);
    asm("# done");
*/
    showstack();
}

int main() {
    f(1, 2, 3, 4, 5, 6, 7, 8,
      0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0);
}
