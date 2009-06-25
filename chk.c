#include <stdio.h>

#define HOGE 42

#define CHK(x) printf(#x "=%d\n", x)

int main() {
    CHK(HOGE);

}
