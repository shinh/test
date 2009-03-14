__attribute__((fastcall)) void f(int x) {
    printf("%d\n", x);
}

int main() {
    f(2);
}
