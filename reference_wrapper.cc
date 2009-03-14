#include <utility>

#define OUT(T) std::reference_wrapper<T>

void int_copy(int i, OUT(int) j) {
    j = i;
}

int main() {
    int x = 1;
    int_copy(42, x);
}
