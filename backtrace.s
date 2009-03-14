.globl backtrace
backtrace:
        push %rbp
        mov %rsp, %rbp
        push %rdi
        push %rsi

        mov %rbp, %rdx
loop:
        cmp $0, %rsi
        je done

        mov 8(%rdx), %rax
        mov %rax, (%rdi)

        mov (%rdx), %rdx
        add $8, %rdi
        dec %rsi

        cmp $0, %rdx
        jne loop
done:
        sub (%rsp), %rsi
        xor %rax, %rax
        sub %rsi, %rax
        pop %rsi
        pop %rdi
        leaveq
        retq
