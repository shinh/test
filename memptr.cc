#include <stdio.h>

class C {
public:
    void f() {}
    virtual void vf() {}
};

int main() {
    printf("%p\n", &C::f);
    printf("%p\n", &C::vf);
}
