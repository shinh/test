#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  symlink(argv[1], argv[2]);
  perror("symlink");
}
