#include <boost/function.hpp>

using namespace boost;

/*
void f() {
    puts("Hello, world!");
}
*/

struct F {
    void operator()() const {
        puts("Hello, world!");
    }
};

int main() {
    //void(*fp)() = &f;
    //fp();
    //function<void()> fp(&f);
    //fp();
    function<void()> fp = F();
    fp();
    //F()();
}
