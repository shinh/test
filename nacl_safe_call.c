#include <stdio.h>

#ifdef __cplusplus
extern "C" {
# define ELLIPSIS ...
#else
# define ELLIPSIS
#endif

static void* g_nacl_call_fn;
void* nacl_call(ELLIPSIS);

#ifdef __cplusplus
}
#endif

__asm__("nacl_call:"
        "pop %eax\n"
        "add $32, %eax\n"
        "push %eax\n"
        "mov $g_nacl_call_fn, %edx\n"
        "mov (%edx), %edx\n"
        "jmp *%edx\n"
        );

#define NACL_CALL(fn, ...) ({                                   \
      register void* r __asm__("eax") __attribute__((unused));  \
      g_nacl_call_fn = (void*)fn;                               \
      nacl_call(__VA_ARGS__);                                   \
      __asm__ __volatile__("nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           "nop;nop;nop;nop;"                   \
                           );                                   \
      r;                                                        \
    })

void naclret();

asm("naclret:\n"
    "pop %eax\n"
    "and $0xffffffe0, %eax\n"
    "jmp *%eax\n");

int f(int argc) {
  NACL_CALL(naclret);
  return (int)NACL_CALL(printf, "%d\n", argc);
}

int main(int argc) {
  return f(argc);
}
