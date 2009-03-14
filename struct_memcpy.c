#include <stdio.h>

typedef struct {
    int i;
    char buf[1000];
    char buf2[1000];
} S;

void init_S(S* s);

int main() {
    S s;
    init_S(&s);
    S s2 = s;
    printf("%s\n", s2.buf);
}

