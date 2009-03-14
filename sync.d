class C {
    void f() {}
}

class D : C {
    synchronized void f() {}
}

void main() {
    C c = new C;
    D d = new D;
    printf("%d %d\n", C.sizeof, D.sizeof);
    synchronized(c) {
    }
    synchronized(d) {
    }
}
