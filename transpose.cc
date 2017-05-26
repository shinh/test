#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <random>
#include <vector>

using namespace std;

constexpr int X = 100;
constexpr int Y = 4;
constexpr int Z = 32000;

float buf[X*Y*Z];
float out[X*Y*Z];

float* Transpose(float* buf) {
  for (int i = 0; i < X; i++) {
    for (int j = 0; j < Y; j++) {
      memcpy(&out[(j*X+i)*Z], &buf[(i*Y+j)*Z], Z * sizeof(float));
    }
  }
  return out;
}

#define BENCH(fn) do {                              \
    clock_t end_clock = clock() + CLOCKS_PER_SEC;   \
    int t = 0;                                      \
    vector<float> result;                           \
    for (; clock() < end_clock; t++) {              \
      fn(buf);                                      \
    }                                               \
    printf("%s %d\n", #fn, t);                      \
  } while (0)

int main() {
  for (int i = 0; i < X*Y; i++) {
    for (int j = 0; j < Z; j++) {
      buf[i*Z+j] = i*Z+j;
    }
  }

  BENCH(Transpose);
}
