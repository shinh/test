template <int i> struct dummy {};

template <class T> int is_defined_helper(dummy<sizeof(&T::hoge) * 0>*);
template <class T> char is_defined_helper(...);

template <class T>
struct is_defined {
    enum { val = sizeof(is_defined_helper<T>(0)) / sizeof(int) };
};

#include <stdio.h>

struct Defined {
    void hoge() {}
};

struct Undefined {
};

int main() {
    printf("%d\n", is_defined<Defined>::val);
    printf("%d\n", is_defined<Undefined>::val);
}
