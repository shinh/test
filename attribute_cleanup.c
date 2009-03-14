#include <stdio.h>

inline void cleanup_int(int *p) {
    *p = 0;
}

struct S {} __attribute__((__cleanup__(cleanup_int)));

typedef int __attribute__((__cleanup__(cleanup_int))) INT;

int f() {
     INT i;
     for (i = 0; i < 42; i++) {
         if (i > 0) {
             INT j = i * 2;
             INT k = i * 3;
             INT l = i * 4;
             INT m = i * 5;
             INT n = i * 6;
             INT o = i * 7;
             INT p = i * 8;
             INT q = i * 9;
             INT r = i * 10;
             INT s = i * 11;
             INT t = i * 12;
             i += j + k * l / m + n * o * q * p / r +s % t;
         }
         printf("%d\n", i);
     }
     printf("%d\n", i);
}

int g() {
    int i;
    printf("%d\n", i);
}

int main() {
    f();
    g();
}
