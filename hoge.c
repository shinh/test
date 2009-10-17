#include <stdio.h>

void hoge() {
    puts("hoge");
}

int main() {
    //hoge();
    fputc(-1, stdout);
}
