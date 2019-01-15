#include <stdio.h>
#include <memory>
#include <utility>

using namespace std;

class C {
public:
    C() { puts("default"); }
    C(const C& c) { puts("copy"); }
    C(C&& c) { puts("move"); }
    C& operator=(const C& c) { puts("assign"); }
    C&& operator=(C&&) { puts("move assign"); }
    ~C() { puts("dtor"); }
};

tuple<C, C> func() {
    C c;
    C c2;
    //return std::tie(c, c2);
    return std::make_tuple(std::move(c), std::move(c2));
    //return std::forward_as_tuple(c, c2);
    //return std::make_tuple(std::move(c), std::move(c2));
}

tuple<C, std::unique_ptr<C>> func2() {
    C c;
    std::unique_ptr<C> cp;
    return std::make_tuple(c, std::move(cp));
    //return std::make_tuple(std::move(c), std::move(c2));
}

int main() {
    puts("func");
    {
        C c, c2;
        std::tie(c, c2) = func();
    }
    puts("func2");
    {
        C c;
        std::unique_ptr<C> cp;
        std::tie(c, cp) = func2();
    }
    puts("func3");
    {
        auto t = func2();
    }
}
