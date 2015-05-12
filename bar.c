void baz() {
}

void bar() {
  asm("push %eax\n"
      "call baz\n"
      "pop %eax\n");
}

int main() {
  bar();
}