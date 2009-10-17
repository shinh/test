#include <stdio.h>
struct C {
    void f() {
        hoge<1> a;
    }
    //int hoge, a;
    template <int i> struct hoge {
        hoge() {
            puts("hoge");
        }
    };
};
int main() {
    C().f();
}
