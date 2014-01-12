int main() {
  va_func(42.0);
  extern void va_func(float f, ...);
  va_func(42.0);
}
