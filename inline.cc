static inline void f(int i) {
    int* p = 0;
    int j ;
    for (j = 0; j < i; j++) {
        p += j;
    }
    for (j = 0; j < i; j++) {
        *p += j;
    }
    for (j = 0; j < i; j++) {
        p += j;
    }
}

int main() {
    f(10);
}
