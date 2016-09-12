import core.stdc.stdio;
import core.vararg;

int vector_dim3;
int vector_dim4;

template Vector(alias dim) {
  class Vector {
    this(...) {
      assert(_arguments.length == dim);
      v = new int[](dim);
      for (int i = 0; i < dim; i++) {
        v[i] = va_arg!int(_argptr);
      }
    }

    int inner_product(Vector!(dim) o) {
      int r = 0;
      for (int i = 0; i < dim; i++) {
        r += v[i] * o.v[i];
      }
      return r;
    }

    int[] v;
  }
}

void main() {
  vector_dim3 = 3;
  vector_dim4 = 4;

  auto v3_1 = new Vector!(vector_dim3)(1, 2, 3);
  auto v3_2 = new Vector!(vector_dim3)(3, 2, 1);
  printf("%d\n", v3_1.inner_product(v3_2));  // OK - 10

  auto v4_1 = new Vector!(vector_dim4)(1, 2, 3, 4);
  auto v4_2 = new Vector!(vector_dim4)(4, 3, 2, 1);
  printf("%d\n", v4_1.inner_product(v4_2));  // OK - 20

  // Not OK! (dimension mismatch) - statically checked
  // printf("%d\n", v3_1.inner_product(v4_1));

  // Not OK! (argument number mismatch) - runtime assertion failure
  // auto v_err = new Vector!(vector_dim3)(1, 2);
}
