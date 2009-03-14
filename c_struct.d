extern(C):
struct S {
    long l;
}
extern(D):
void main() {
    S s;
    printf("%d\n", s.l.sizeof);
}
