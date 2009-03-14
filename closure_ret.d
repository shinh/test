void delegate() get_closure(int i) {
    return {printf("%d\n", i);};
}

void invoke_closure(void delegate() dg) {
    //printf("sizeof=%d\n", dg.sizeof);
    dg();
}

void call_closure(int i, int j) {
    printf("%d\n", j);
    invoke_closure({printf("%d %d\n", i, j);});
}

void main() {
    auto c = get_closure(3);
    c();
    auto c2 = get_closure(5);
    c();
    c2();
    call_closure(9, 11);
}
