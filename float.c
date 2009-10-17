void f(int i, float f, int j) {
}

void f2(float f) {
}

float f3(float f) {
    f * 2.3;
}

void d(int i, double f, int j) {
}

void d2(double f) {
}

double d3(double f) {
    f * 2.3;
}

int main() {
    asm("#\n");
    d3(2.4);
    asm("#\n");
    d2(2.4);
    d(1, 1.2, 2);

    f3(2.4);
    f2(2.4);
    f(1, 1.2, 2);
}
