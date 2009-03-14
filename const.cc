struct S {
    S() {}
    void f() {}
};

void f() {
    const S s;
    s.f();
}
