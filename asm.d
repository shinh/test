void main() {
    int[2] i;
    int* p = i.ptr;
    ulong pv = cast(ulong)p;
    char c;
    asm {
        mov EAX, c;
//        mov RAX, pv;
        mov 4[RAX], EBP;
    }
    printf("%d\n", i[0], i[1]);
}
