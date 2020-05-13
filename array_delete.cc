#include <stdio.h>

class C {
public:
    ~C() {
        puts("hoge");
    }
};

int main() {
    C* c = new C[10];
    delete[] c;
}
