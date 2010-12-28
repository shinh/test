#include "raw_write.h"

#include <stdlib.h>
#include <string.h>

int main() {
    RAW_WRITE(2, "hello\n", 6);
    RAW_PRINT_STR("hello world!\n");
    RAW_PUTS_INT(3000);
    RAW_PUTS_INT(0);
    {
        long long ll = -1152921504606846976;
        RAW_PUTS_INT(ll);
        RAW_PUTS_HEX(ll);
    }
    {
        char *p = malloc(1000000);
        strcpy(p, "malloc-ed buffer");
        RAW_PUTS_STR(p);
    }
    RAW_NOP();
    RAW_UNIQ_NOP();
    RAW_BREAK();
}
