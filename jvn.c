#include <stdio.h>
#include <limits.h>
#include <stdint.h>
int main(int len, char* argv[]) {
    //len -= 3;
    //len = INT_MAX;
    int l = INT_MAX;
    //unsigned int l = -1;
    if (l + argv[0] < argv[0]) {
        printf("hoge\n");
    }
}
