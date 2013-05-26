#include <stdio.h>
#include "tls_extern.h"
int ret_tls() {
  return tls;
}
int main() {
  char* code = (char*)tls_use;
  FILE* fp = fopen("tls_code", "wb");
  int i;
  for (i = 0; code[i] != '\xc3'; i++) {
    fputc(code[i], fp);
  }
  return tls_use() + ret_tls();
}
