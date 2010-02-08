#include <stdio.h>

#define snprintf _snprintf

int main() {
    char buf[256];
    memset(buf, 0, sizeof(buf));
    printf("%d\n", snprintf(buf, 2, "hoge"));
    memset(buf, 0, sizeof(buf));
    snprintf(buf, 2, "%d", 123);
    printf("%s\n", buf);

    printf("%d\n", snprintf(buf, 2, "%d", 123));
    printf("%d\n", snprintf(buf, 3, "%d", 123));
    printf("%d\n", snprintf(buf, 4, "%d", 123));
}
