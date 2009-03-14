#include <stdio.h>
int main() {
    int i;
    goto i;
    for (i = 0; i < 3; i++) {
        break;
    i:  printf("%d\n", i);
    }
}
