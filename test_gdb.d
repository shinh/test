class C {
    this(int x) {
        x_ = x;
    }
    int x_;
}

void f() {
    printf("Hello, world!\n");
    asm {
        int 3;
    }
}

void main() {
    string a = "hoge";
    int [] ia = [ 2, 3, 4 ];
    C c = new C(3);
    f();
}
