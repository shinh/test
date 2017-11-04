typedef unsigned long ulong;

ulong fib_slow(ulong n) {
  if (n <= 1) {
    return 1;
  } else {
    return fib_slow(n-1) + fib_slow(n-2);
  }
}

int main() {
  printf("%lld\n", fib_slow(46));
}
