int main(int argc, char* argv[]) {
    long long v = atoi(argv[1]);
    v *= v;
    long long s = sqrt(v+0.00001);
    if (s * s != v) {
        puts("Fail");
    }
}
