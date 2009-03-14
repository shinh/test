#include <stdio.h>

typedef unsigned int uint;

uint temper(uint y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

uint calc(uint a, uint b) {
    int t;

    a ^= (a >> 11);
    a ^= (a << 7) & 0x9d2c5680UL;
    a ^= (a << 15) & 0xefc60000UL;

    b ^= (b >> 11);
    b ^= (b << 7) & 0x9d2c5680UL;
    b ^= (b << 15) & 0xefc60000UL;

    a ^= (a >> 18);
    b ^= (b >> 18);

    t = a ^ b;
    b = (a & b) << 1;
    a = t;
    t = a ^ b;
    b = (a & b) << 1;
    a = t;
    t = a ^ b;
    b = (a & b) << 1;
    a = t;
    t = a ^ b;
    b = (a & b) << 1;
    a = t;

    return a + b;
}

uint test(uint a, uint b) {
    printf("%u %u\n", temper(a)+temper(b), calc(a, b));
}

int main() {
    test(123456789, 234567890);
    test(123456789, 23456789);
    test(123456789, 1);
    test(1, 234567890);
}
