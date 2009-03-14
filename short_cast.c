#include <stdio.h>
int main() {
    int var1 = 0x1020304;
    *(short *)&var1 = 0x0809;
    printf("var1=%x\n", var1);
};
