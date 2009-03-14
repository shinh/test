enum {
  A
};

template <class T> void f(T t) {
};

void f(int t) {
}

int main() {
  f(A);
}

