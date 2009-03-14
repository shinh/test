#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main() {
    static const int N = 500 * 1024 * 1024;
    char* src = (char*)malloc(N);
    char* dst = (char*)malloc(N);
    clock_t s = clock();
    //memcpy(dst, src, N);

/*
    int i;
    for (i = 0; i < N; i++) *dst++ = *src++;
*/

/*
    int i;
    for (i = 0; i < N/8; i++) {
        asm("movq (%0), %%mm0;"
            "movntq %%mm0, (%1);"
            :: "r"(src), "r"(dst));
        src+=8;
        dst+=8;
    }
*/

    int i;
    for (i = 0; i < N/32; i++) {
        asm("prefetchnta (%0);"
            "prefetchnta 64(%0);"
            "prefetchnta 128(%0);"
            "prefetchnta 192(%0);"
            "prefetchnta 256(%0);"
            "movq (%0), %%mm0;"
            "movntq %%mm0, (%1);"
            "movq 8(%0), %%mm0;"
            "movntq %%mm0, 8(%1);"
            "movq 16(%0), %%mm0;"
            "movntq %%mm0, 16(%1);"
            "movq 24(%0), %%mm0;"
            "movntq %%mm0, 24(%1);"
            :: "r"(src), "r"(dst));
        src+=32;
        dst+=32;
    }

    printf("%f\n", ((double)clock() - s) / CLOCKS_PER_SEC);
}
