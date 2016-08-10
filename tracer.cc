#include <stdint.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__apple__)
#define PTRACE_TRACEME PT_TRACE_ME
#define PT_STEP PTRACE_SINGLESTEP
#endif

typedef struct {
#ifdef __x86_64__
  #define BP rbp
  #define SP rsp
  #define IP rip
  uint64_t  r15,r14,r13,r12,rbp,rbx,r11,r10;
  uint64_t  r9,r8,rax,rcx,rdx,rsi,rdi,orig_rax;
  uint64_t  rip,cs,eflags;
  uint64_t  rsp,ss;
  uint64_t  fs_base, gs_base;
  uint64_t  ds,es,fs,gs;
#else
  #define BP ebp
  #define SP esp
  #define IP eip
  uint32_t  ebx, ecx, edx, esi, edi, ebp, eax;
  uint16_t  ds, __ds, es, __es;
  uint16_t  fs, __fs, gs, __gs;
  uint32_t  orig_eax, eip;
  uint16_t  cs, __cs;
  uint32_t  eflags, esp;
  uint16_t  ss, __ss;
#endif
} Regs;

long checked_ptrace(enum __ptrace_request req, pid_t pid, long addr, long data,
                    const char* req_str, const char* file, int line) {
  long r = ptrace(req, pid, addr, data);
  if (r == -1) {
    fprintf(stderr, "%s:%d: ptrace(%s): ", file, line, req_str);
    perror("");
    //exit(1);
  }
  return r;
}

#define CHECKED_PTRACE(req, pid, addr, data)    \
  checked_ptrace(req, pid, addr, data, #req, __FILE__, __LINE__)

int main(int argc, char* argv[]) {
  argc--;
  argv++;
  pid_t pid = fork();
  if (pid) {
    long prev = 0;
    while (true) {
      int status;
      wait(&status);
      if (!WIFSTOPPED(status))
        break;

      Regs regs;
      CHECKED_PTRACE(PTRACE_GETREGS, pid, 0, (long)&regs);
      if (prev != regs.IP)
        printf("%p\n", regs.IP);
      prev = regs.IP;

      CHECKED_PTRACE(PTRACE_SINGLESTEP, pid, 0, 0);
    }
  } else {
    CHECKED_PTRACE(PTRACE_TRACEME, 0, 0, 0);
    execv(argv[0], argv);
  }
}
