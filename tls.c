__thread int x;
int* f() {
  return &x;
}
