#ifdef __i386__
# define RDTSC_REG "=A"
#else
# define RDTSC_REG "=a"
#endif

#define RDTSC(X)                                                        \
    do {                                                                \
        __asm__ __volatile__ ("push %%ebx;\n"                           \
                              "cpuid;\n"                                \
                              "pop %%ebx;\n"                            \
                              ::: "eax", "ecx", "edx");                 \
        __asm__ __volatile__ ("rdtsc": RDTSC_REG(X));                   \
    } while (0);

unsigned long long rdtsc() {
    unsigned long long r;
    RDTSC(r);
    return r;
}

#include <stdio.h>

int main() {
    int i;
    for (i = 0; i < 3; i++) {
        unsigned long long st = rdtsc();
        unsigned long long ed = rdtsc();
        printf("%lld\n", ed - st);
    }
    return 0;
}
