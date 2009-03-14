#include <stdlib.h>
int main() {
    const int N = 250000000;
    int* p = (int*)malloc(N*4);
    int i;
    for (i = 0; i < N; i++) {
        p[i] = i;
    }
    --*"";
}
