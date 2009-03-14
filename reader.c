#include <stdio.h>
#include <unistd.h>

int main() {
    char buf[4096];
    while (1) {
        read(0, buf, 100);
        usleep(1000);
    }
}
