#include <stdio.h>
int main() {
    int i = 0;
    asm("mov $2, %%eax;"
        "mov $-1, %%edx;"
        "add %%edx, %%eax;"
        "setc %0;" :"=m"(i));
    printf("%d\n", i);
}
