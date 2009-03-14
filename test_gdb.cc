#include <vector>

void f() {
    printf("Hello, world!\n");
    asm("int $3");
}

int main() {
    int ia[] = {2,3,4};
    char a[] = "hoge";
    std::vector<int> iv;
    iv.push_back(2);
    iv.push_back(3);
    iv.push_back(4);
    f();
}
