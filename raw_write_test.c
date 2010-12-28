#include "raw_write.h"

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
    RAW_NOP();
    RAW_UNIQ_NOP();
    RAW_BREAK();
}
