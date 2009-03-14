#include <stdio.h>

int main() {
    char buf[256];
    printf("%d\n", snprintf(buf, 2, "hoge"));
}
