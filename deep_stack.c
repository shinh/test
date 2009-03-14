int* f() {
    int x = 3;
    return &x;
}

int main() {
    int* p = f();
    int x = *p;
    return x;
}
