extern(C++):
interface C {
    void f();
};

extern(D):
void main() {
    C c = new C;
    c.f();
}
