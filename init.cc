#include <stdio.h>

struct S {
    int i;
};

void print_s_without_paren() {
    void* i;
    (&i)[-1] = (void*)42;
    S s;
    printf("%d\n", s.i);
}

void print_s_with_paren() {
    void* i;
    (&i)[-1] = (void*)42;
    S s = S();
    //const S& s = S();
    printf("%d\n", s.i);
}

int main() {
    print_s_without_paren();
    print_s_with_paren();
}
