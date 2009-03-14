.hello:
        .string   "Hello, world!"
.format:
        .string   "%s\n"
foo:    
;;         push    %ebp
;;         mov     %esp, %ebp
;        push    %ecx
        movl    $.hello, -0x4(%ebp)
;;         mov     %ebp, %esp
;;         pop     %ebp
        ret
bar:
;;         push    %ebp
;;         mov     %esp, %ebp
;        push    %ecx
        mov     -0x4(%ebp), %eax
        push    %eax
        push    $.format
        call    printf
;;         add     $0x8, %esp
;;         mov     %ebp, %esp
;;         pop     %ebp
        ret
.globl  main
main:
        push    %ebp
        mov     %esp, %ebp

        call    foo
        call    bar
        xor     %eax, %eax
        pop     %ebp
        ret
