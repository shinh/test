.globl main
main:
        mov $1, %eax
        mov $2, %ebx
        sysenter
        #int $0x80
        #mov $60, %rax
        #mov $2, %rdi
        #syscall
