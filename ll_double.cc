int main() {
    long long v = 99999787;
    v *= v;
    long long s = sqrt(v+0.00001);
    if (s * s != v) {
        puts("Fail");
    }
}
