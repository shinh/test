#include <stdio.h>
#include <setjmp.h>

jmp_buf env;

void g() {
    longjmp(env, 1);
}

class C {
public:
    ~C() {
        puts("~C()");
    }
};

void f() {
    C c;
}

int main() {
    if (!setjmp(env)) {
        f();
    }
}
