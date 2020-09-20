#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

template <typename T>
T align_up(T v, int align, int off = 0) {
    return (T)((((uintptr_t)v + align - 1) & ~(align - 1)) + off);
}

int main() {
    float* p = (float*)malloc(10000);
    p = align_up(p, 32, 16);
    __m256 v;
    v = _mm256_loadu_ps(p);
    v = _mm256_load_ps(p);
}

// $ g++ -mavx2 simd_align.cc && ./a.out
// zsh: segmentation fault  ./a.out

// (gdb) disassemble main
// Dump of assembler code for function main:
// 0x0000555555555775 <+0>:     push   %rbp
// 0x0000555555555776 <+1>:     mov    %rsp,%rbp
// 0x0000555555555779 <+4>:     and    $0xffffffffffffffe0,%rsp
// 0x000055555555577d <+8>:     sub    $0x40,%rsp
// 0x0000555555555781 <+12>:    mov    $0x2710,%edi
// 0x0000555555555786 <+17>:    callq  0x5555555558e0 <malloc@plt>
// 0x000055555555578b <+22>:    mov    %rax,0x8(%rsp)
// 0x0000555555555790 <+27>:    mov    0x8(%rsp),%rax
// 0x0000555555555795 <+32>:    mov    $0x10,%edx
// 0x000055555555579a <+37>:    mov    $0x20,%esi
// 0x000055555555579f <+42>:    mov    %rax,%rdi
// 0x00005555555557a2 <+45>:    callq  0x5555555557ec <_Z8align_upIPfET_S1_ii>
// 0x00005555555557a7 <+50>:    mov    %rax,0x8(%rsp)
// 0x00005555555557ac <+55>:    mov    0x8(%rsp),%rax
// 0x00005555555557b1 <+60>:    mov    %rax,0x18(%rsp)
// 0x00005555555557b6 <+65>:    mov    0x18(%rsp),%rax
// 0x00005555555557bb <+70>:    vmovups (%rax),%xmm0
// 0x00005555555557bf <+74>:    vinsertf128 $0x1,0x10(%rax),%ymm0,%ymm0
// 0x00005555555557c6 <+81>:    vmovaps %ymm0,0x20(%rsp)
// 0x00005555555557cc <+87>:    mov    0x8(%rsp),%rax
// 0x00005555555557d1 <+92>:    mov    %rax,0x10(%rsp)
// 0x00005555555557d6 <+97>:    mov    0x10(%rsp),%rax
// => 0x00005555555557db <+102>:   vmovaps (%rax),%ymm0
// 0x00005555555557df <+106>:   vmovaps %ymm0,0x20(%rsp)
// 0x00005555555557e5 <+112>:   mov    $0x0,%eax
// 0x00005555555557ea <+117>:   leaveq
// 0x00005555555557eb <+118>:   retq
