class C {
public:
    typedef int C::*mv;
    typedef int (C::*mf)();

    int f() {}
    int v;

    mf mvf();
    mv mvv();
};

C::mf C::mvf() { return &C::f; }
C::mv C::mvv() { return &C::v; }
