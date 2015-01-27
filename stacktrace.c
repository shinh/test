#include <stdio.h>

typedef struct {
  void* bp;
  void* ret;
} frame;

void print_trace() {
  frame* fp = __builtin_frame_address(0);
  while (fp != NULL) {
    printf("%p\n", fp->ret);
    fp = (frame*)fp->bp;
  }
}

int main() {
  print_trace();
}
