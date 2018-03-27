struct ct_nil {};

template <int V, class Rest>
struct ct_cons {
  static const int val = V;
  typedef Rest rest;
};

template <class List>
struct ct_sum {
  static const int result = List::val + ct_sum<typename List::rest>::result;
};

template <>
struct ct_sum<ct_nil> {
  static const int result = 0;
};

typedef ct_cons<4, ct_cons<3, ct_cons<7, ct_nil> > > my_list;
const int result = ct_sum<my_list>::result;

#include <stdio.h>

int main() {
  printf("%d\n", result);
}
