#include <stdio.h>
#include <stdlib.h>
int main(int argc)
{
    int i, j, sum = 0;
    int x = rand()%3+3;
    int y = argc + 10;
    for (i=0; i<10000000; i++) {
    for (j=0; j<10; j++) {
#if 0
        div_t d;
        d = div(x, y);
        sum += d.quot;
        sum += d.rem;
#else
        sum += x / y;
        sum += x % y;
#endif
        x++; y++;
    }
    }
    printf("%d\n", sum);
    return 0;
}
