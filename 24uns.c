#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    int i;
    char* buf = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24";
    unsigned hits[24];
#if 0
    if (sscanf(buf,
               "%u,%u,%u,%u,%u," "%u,%u,%u,%u,%u,"
               "%u,%u,%u,%u,%u," "%u,%u,%u,%u,%u,"
               "%u,%u,%u,%u",
               hits +  0, hits +  1, hits +  2, hits +  3, hits +  4,
               hits +  5, hits +  6, hits +  7, hits +  8, hits +  9,
               hits + 10, hits + 11, hits + 12, hits + 13, hits + 14,
               hits + 15, hits + 16, hits + 17, hits + 18, hits + 19,
               hits + 20, hits + 21, hits + 22, hits + 23)
        != 24) {
        return -1;
    }
#else
    char* p;
    for (p = buf, i = 0;; i++) {
        hits[i] = strtol(p, &p, 10);
        if (!*p) break;
        assert(*p == ',');
        p++;
    }
    assert(i == 23);
#endif

    for (i = 0; i < 24; i++) {
        assert(i+1 == hits[i]);
    }
}
