#include <stdio.h>

struct Foo {
    int i;
};

void with_paren() {
    Foo* foo = new Foo;
    foo->i = 42;
    delete foo;
    foo = new Foo;
    printf("with paren: %d\n", foo->i);
}

void without_paren() {
    Foo* foo = new Foo;
    foo->i = 42;
    delete foo;
    foo = new Foo();
    printf("without paren: %d\n", foo->i);
}

int main() {
    with_paren();
    without_paren();
}
