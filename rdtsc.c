#define RDTSC(X)                                                        \
    do {                                                                \
        unsigned int eax, edx;                                          \
        __asm__ __volatile__ ("cpuid"::: "eax", "ebx", "ecx", "edx");   \
        __asm__ __volatile__ ("rdtsc": "=a"(eax), "=d"(edx));           \
        X = ((unsigned long long)edx << 32) | eax;                      \
    } while (0);

unsigned long long rdtsc() {
    unsigned long long r;
    RDTSC(r);
    return r;
}

#include <stdio.h>

int main() {
    unsigned long long st = rdtsc();
    unsigned long long ed = rdtsc();
    printf("%lld\n", ed - st);
    return 0;
}
