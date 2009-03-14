int fact(int x) {
    return x == 1 ? 1 : fact(x-1) * x;
}
