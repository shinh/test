#include <windows.h>
#include <stdio.h>

int main() {
  char buf[256];
  gethostname(buf, 255);
  printf("%s\n", buf);
}
