#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif
void* nacl_call(void* fn, ...);

__asm__("nacl_call:"
        "pop %eax\n"
        "pop %edx\n"
        "add $32, %eax\n"
        "push %eax\n"
        "jmp *%edx\n");

#define NACL_CALL(fn, ...) ({                   \
      register void* r __asm__("eax");          \
      nacl_call((void*)fn, ##  __VA_ARGS__);    \
      __asm__ __volatile__("nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           "nop;nop;nop;nop;"   \
                           );                   \
      r;                                        \
    })

int main(int argc) {
  return (int)NACL_CALL(printf, "%d\n", argc);
}
