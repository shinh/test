#include <stdio.h>
#include <stdlib.h>

# include <sys/mman.h>

typedef int(*adder_t)(int);

int add(int x, int y) {
    return x + y;
}

adder_t make_adder(int i) {
    int* p;
    char* buf = (char*)malloc(18);
    //   0:   67 ff 34 24             addr32 pushq (%esp)
    buf[0] = 0x67;
    buf[1] = 0xff;
    buf[2] = 0x34;
    buf[3] = 0x24;
    //   4:   67 c7 44 24 04 03 00    addr32 movl $0x3,0x4(%esp)
    //   b:   00 00
    buf[4] = 0x67;
    buf[5] = 0xc7;
    buf[6] = 0x44;
    buf[7] = 0x24;
    buf[8] = 0x04;
    p = (int*)(buf + 9);
    *p = i;
    //   d:   e9 00 00 00 00          jmpq   0x12
    buf[13] = 0xe9;
    p = (int*)(buf + 9);
    *p = (int)&add;

    mprotect((void*)((int)p & ~4095),
             4096,
             PROT_READ | PROT_WRITE | PROT_EXEC);
    return (adder_t)p;
}

int main() {
    adder_t adder = make_adder(3);
    printf("%d\n", adder(6));
}
