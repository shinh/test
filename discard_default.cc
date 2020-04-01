int FMA(int a, int b, int c = 4) {
    return a * b + c;
}

#define DISCARD_DEFAULT_2(fn) [](auto&& a, auto&& b) { return FMA(a, b); }

int main() {
    auto fp = FMA;
    fp(2, 3, 5);
    auto fp2 = DISCARD_DEFAULT_2(FMA);
    fp2(2, 3);
}
