#include <stdio.h>
int main() {
    printf("%d\n", ((1 - sizeof(int)) >> 32));
}
