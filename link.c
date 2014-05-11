#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  link(argv[1], argv[2]);
  perror("link");
}
