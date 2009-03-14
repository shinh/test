.globl  leap
leap:
        mov 4(%esp), %eax
        mov $100, %ecx
        xor %edx, %edx
        idiv %ecx
        cmp $0, %edx
        jne .l
        mov %eax, %edx
.l:
        and $3, %edx
        sete %al
        ret
