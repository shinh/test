#include <stdio.h>
#include <unistd.h>
#include <sys/select.h>

#define N 4096

int main() {
    char buf[N];
    int i;
    for (i = 0; i < N; i++) {
        buf[i] = 'h';
    }
    int n = 0;
    while(1) {
        fd_set fdset;
        FD_ZERO(&fdset);
        FD_SET(1, &fdset);
        fprintf(stderr, "select\n");
        select(2, NULL, &fdset, NULL, NULL);
        fprintf(stderr, "write\n");
        int sz = write(1, buf, N);
        fprintf(stderr, "%d %d\n", n, sz);
        n++;
    }
}
