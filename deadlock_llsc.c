#include <stdio.h>

int main() {
    int i, j;
    int* ip = &i;
    int* jp = &j;
    asm(".lock_cnt:\n"
        " lwarx 7, 0, %0;\n"
        ".lock_cnt2:\n"
        " lwarx 8, 0, %1;\n"
        " stwcx. 8, 0, %1;\n"
        " bne- .lock_cnt2;\n"
        " stwcx. 7, 0, %0;\n"
        " bne- .lock_cnt;\n"
        :"=&r"(ip),"=&r"(jp)::"7","8");
}
