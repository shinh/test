#include <stdio.h>

template<typename T>
struct X {
    friend int f(const T&, const T&) { return 3; }
};

template<int i>
struct I {
    friend void g() { printf("%d\n", i); }
};

//struct Foo : X<Foo> {}; // (A)

//struct Foo{};
//template struct X<Foo>;

struct Foo {};
struct Bar : X<Foo> {};

int main()
{
    //I<1> i1;
    I<3> i3;
    g();

    Foo a, b;
    return f(a, b);
}
