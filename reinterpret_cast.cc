#include <stdio.h>
int main() {
    int i = 1000000000;
    float f = (float)i;
    float f2 = *(float*)&i;
    printf("%f %f\n", f, f2);
}
