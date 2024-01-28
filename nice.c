#include <stdio.h>
#include <unistd.h>

int main() {
    printf("nice(1) %d\n", nice(1));
    printf("nice(-1) %d\n", nice(-1));
}
