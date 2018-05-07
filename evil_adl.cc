#include <stdio.h>

namespace A {
  struct A {};
}
namespace B {
  struct B {
    B() {}
    B(const A::A&) {}
    operator A::A() { return A::A(); }
  };
}

namespace A {
  void ok(A a) {
    puts("in A");
  }

  void ambiguous(A::A a, B::B b) {
    puts("in A");
  }
}

namespace B {
  void ok(B::B b) {
    puts("in B");
  }

  void ambiguous(A::A a, B::B b) {
    puts("in B");
  }
}

int main() {
  A::A a;
  B::B b;
  ok(a);  // in A
  ok(b);  // in B
  B::ok(a);  // in B
  A::ok(b);  // in A
  //ambiguous(a, b);
  A::ambiguous(a, b);  // in A
}
