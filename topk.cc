#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <random>
#include <vector>

using namespace std;

static const unsigned int K = 4;
static const int kNumData = 320000;

vector<float> topk_sort(const vector<float>& data) {
  vector<float> sorted(data);
  sort(sorted.begin(), sorted.end());
  sorted.resize(K);
  return sorted;
}

vector<float> topk_nth(const vector<float>& data) {
  vector<float> topk;
  for (size_t i = 0; i < K; i++)
    topk.push_back(data[i]);
  sort(topk.begin(), topk.end());

  for (size_t i = K; i < data.size(); i++) {
    float v = data[i];
    if (topk[K-1] < v)
      continue;
    topk.push_back(v);
    if (topk.size() > 2 * K) {
      nth_element(topk.begin(), topk.begin() + K - 1, topk.end());
      topk.resize(K);
    }
  }

  sort(topk.begin(), topk.end());
  topk.resize(K);
  return topk;
}

vector<float> topk_heap(const vector<float>& data) {
  vector<float> topk;
  for (size_t i = 0; i < K; i++)
    topk.push_back(data[i]);
  make_heap(topk.begin(), topk.end());

  for (size_t i = K; i < data.size(); i++) {
    float v = data[i];
    if (topk[0] < v)
      continue;
    topk.push_back(v);
    push_heap(topk.begin(), topk.end());
    pop_heap(topk.begin(), topk.end());
    topk.pop_back();
  }

  sort(topk.begin(), topk.end());
  topk.resize(K);
  return topk;
}

vector<float> topk_avx_heap(const vector<float>& data) {
  vector<float> topk;
  const size_t aligned_k = (K + 7) & ~7;
  for (size_t i = 0; i < aligned_k; i++)
    topk.push_back(data[i]);
  sort(topk.begin(), topk.end());
  topk.resize(K);

  make_heap(topk.begin(), topk.end());
  __m256 worst = _mm256_set1_ps(topk[0]);
  for (size_t i = aligned_k; i < data.size(); i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    __m256 mask = _mm256_cmp_ps(worst, v, _CMP_GE_OQ);
    if (!(_mm256_movemask_ps(mask) & 255))
      continue;
    float buf[8] __attribute__((aligned(32)));
    _mm256_store_ps(buf, v);
    for (int j = 0; j < 8; j++) {
      float v = buf[j];
      if (topk[0] < v)
        continue;
      topk.push_back(v);
      push_heap(topk.begin(), topk.end());
      pop_heap(topk.begin(), topk.end());
      topk.pop_back();
    }
    worst = _mm256_set1_ps(topk[0]);
  }

  sort(topk.begin(), topk.end());
  topk.resize(K);
  return topk;
}

vector<float> topk_avx_nth(const vector<float>& data) {
  vector<float> topk;
  const size_t aligned_k = (K + 7) & ~7;
  for (size_t i = 0; i < aligned_k; i++)
    topk.push_back(data[i]);
  sort(topk.begin(), topk.end());
  topk.resize(K);

  __m256 worst = _mm256_set1_ps(topk[K-1]);
  for (size_t i = aligned_k; i < data.size(); i += 8) {
    __m256 v = _mm256_loadu_ps(&data[i]);
    __m256 mask = _mm256_cmp_ps(worst, v, _CMP_GE_OQ);
    if (!(_mm256_movemask_ps(mask) & 255))
      continue;
    float buf[8] __attribute__((aligned(32)));
    _mm256_store_ps(buf, v);
    for (int j = 0; j < 8; j++) {
      float v = buf[j];
      if (topk[0] < v)
        continue;
      topk.push_back(v);
    }
    if (topk.size() > 2 * K) {
      nth_element(topk.begin(), topk.begin() + K - 1, topk.end());
      topk.resize(K);
      worst = _mm256_set1_ps(topk[K-1]);
    }
  }

  sort(topk.begin(), topk.end());
  topk.resize(K);
  return topk;
}

#define BENCH(fn) do {                              \
    clock_t end_clock = clock() + CLOCKS_PER_SEC;   \
    int t = 0;                                      \
    vector<float> result;                           \
    for (; clock() < end_clock; t++) {              \
      result = fn(data);                            \
    }                                               \
    printf("%s %d\n", #fn, t);                      \
    assert(expected == result);                     \
  } while (0)

int main() {
  mt19937 mt = mt19937(random_device()());
  vector<float> data;
  for (int i = 0; i < kNumData; i++)
    data.push_back(mt());

  vector<float> expected = topk_sort(data);

  BENCH(topk_avx_heap);
  BENCH(topk_avx_nth);
  BENCH(topk_nth);
  BENCH(topk_heap);
  BENCH(topk_sort);
}
