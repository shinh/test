#include <ext/vstring.h>

using namespace __gnu_cxx;

int main() {
    __vstring s("hoge");
    printf("%d %s\n", sizeof(__vstring), s.c_str());
}
