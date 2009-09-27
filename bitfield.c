#include <stdio.h>

typedef struct {
    unsigned long long a: 1;
    unsigned long long ll: 50;
    unsigned int r;
} S;

typedef struct {
    unsigned int a: 23;
    unsigned int b: 1;
    unsigned int c: 1;
    unsigned int r;
} S2;

void dump_bits(unsigned long long ll) {
    int n = 1;
    unsigned long long t = ll;
    while (t) {
        n++;
        t >>= 1;
    }
    while (n) {
        printf("%d", (ll >> n) & 1);
        n--;
    }
    puts("");
}

int main() {
/*
    S2 s2;
    s2.a = (1 << 23) - 1;
    s2.b = 1;
    s2.c = 1;
    s2.r = 0;
    dump_bits(*(unsigned long long*)&s2);
    printf("%d\n", s2.a);
*/

    S s;
    //s.a = 0;
    //s.ll = (1ULL << 50) -  1;
    s.ll = 0x123456789aULL;
    printf("%llx\n", s.ll);
    printf("%llx\n", s.ll++);
    s.r = 0;
    //s.ll = 0x123456789a;
    //s.r = 0;

    //printf("%llu\n", 0xffffffffffffffffULL);
    //printf("%llu\n", 0xffffffffffffULL);
    dump_bits(s.ll);
    printf("%llx\n", s.ll);
    dump_bits(*(unsigned long long*)&s);
}
