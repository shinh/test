.globl  main
main:
        mov     $.L, %rdi
        mov     $-1, %rsi
        sar     $2, %rsi
        xor     %eax, %eax
        call    printf
        ret
.L:     .string "%d\n"
