#include <stdio.h>

int main() {
    int a;
    struct sbf2 {
        long long f1 : 3;
        long long : 2;
        long long f2 : 35;
        long long : 0;
        long long f3 : 45;
        long long f4 : 7;
        unsigned long long f5 : 7;
    } st2;
    st2.f1 = 3;
    st2.f2 = 0x123456789ULL;
    st2.f3 = 15;
    a = 120;
    st2.f4 = (long long)a << 43;
    st2.f5 = a;
    st2.f5++;
    printf("%lld %lld %lld %lld %lld\n",
           st2.f1, st2.f2, st2.f3, st2.f4, st2.f5);
}
