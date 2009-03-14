import std.c.stdio;

void f(int delegate(int x) dg) {
    printf("%d\n", dg(3));
}

void main() {
    void delegate() fun = {puts("Hello, world!");};
    fun();
    //f((int x){x+1});
}
