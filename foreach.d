class C {
    void f() {}
}

void main() {
    C[] cs;
    cs ~= new C();
    foreach (C c; cs) {
        cs ~= new C();
    }
}