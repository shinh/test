#include <stdio.h>

#include <typeinfo>

#include <cxxabi.h>

using namespace std;

struct C {
    virtual void func() { puts("C"); }
};

struct D : public C {
    virtual void func() { puts("D"); }
};

int main() {
    C* c = new D();
    c->func();  // D

    void (C::*mp)() = &C::func;
    (c->*mp)();  // D::func

    puts("typeinfo");
    void** vtable = *(void***)c;
    std::type_info* ti = (std::type_info*)vtable[-1];
    printf("%d\n", &typeid(*c) == ti);
    printf("%d\n", &typeid(C) == ti);
    printf("%d\n", &typeid(D) == ti);

    // 0x400ca0
    using namespace __cxxabiv1;
    printf("%p\n", dynamic_cast<__si_class_type_info*>(ti));

    // (nil)
    printf("%p\n", dynamic_cast<D*>(new C()));
    // 0xb13050
    printf("%p\n", dynamic_cast<D*>(new D()));

    printf("%d\n", *(int*)&mp);

    printf("%p\n", *(void**)c);
    printf("%p\n", (*(void***)c)[0]);
    printf("%p\n", (*(void***)c)[-1]);

    printf("%d\n", sizeof(D));
}
