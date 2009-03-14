void f() {return 0;}

static if (is(ReturnType!(f) == void)) {
    const int x=1;
} else {
    const int x=2;
}

void main() {
    printf("%d\n", x);
}
