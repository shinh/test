#include <stdio.h>
#include <memory.h>
int foobar(unsigned char* ptr, int len)
{
    int i;
    unsigned char* p = ptr;
    for (i = 0; *p != 0 && i < len; i++, p++);
    return p - ptr;
}
int main(int argc, char* argv[])
{
    int i;
    unsigned char* p;
    for (i = 16300;; i++) {
        p = (unsigned char*)malloc(i);
        memset(p, '\xff', i);
        printf("len = %d\n", foobar(p, i));
        free(p);
    }
}
