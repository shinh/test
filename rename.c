#include <stdio.h>
int main(int argc, const char* argv[]) {
    int r = rename(argv[1], argv[2]);
    if (r < 0) {
        perror("rename");
    }
}