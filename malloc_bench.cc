// g++ -std=c++11 -O -g malloc_bench.cc
//
// use malloc_log.cc
// sed '/^MLOG/!d; s/^MLOG: //; s/ =>//; s/(\(.*\))/ \1/; s/,/ /' log > mlog

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <unordered_map>
#include <vector>

#include "malloc_bench.h"

using namespace std;

#define CHECK(c) do {                           \
    if (!(c))                                   \
      assert(c);                                \
  } while (0)

enum {
  MALLOC,
  FREE,
  CALLOC,
  REALLOC,
};

struct Log {
  int type;
  uintptr_t a0;
  uintptr_t a1;
  uintptr_t r;
};

int get_type(const char* str) {
  if (!strcmp(str, "malloc"))
    return MALLOC;
  if (!strcmp(str, "free"))
    return FREE;
  if (!strcmp(str, "calloc"))
    return CALLOC;
  if (!strcmp(str, "realloc"))
    return REALLOC;
  CHECK(false);
}

vector<void*> ptrs;
unordered_map<void*, uintptr_t> ptr2id;
vector<Log> logs;
size_t malloc_total, malloc_cnt;

void record_alloc(void* p, uintptr_t sz, Log* l) {
  CHECK(p);
  l->r = ptrs.size();
  CHECK(ptr2id.emplace(p, l->r).second);
  ptrs.push_back(p);
  malloc_total += sz;
  malloc_cnt++;
}

void record_free(void* p, Log* l) {
  if (!p)
    return;
  l->a0 = ptr2id[p];
  CHECK(l->a0);
  ptr2id.erase(p);
}

#define DEFINE_RUN_BENCH(n)                                     \
  void run_bench_ ## n() {                                      \
    for (const Log& log : logs) {                               \
      switch (log.type) {                                       \
        case MALLOC:                                            \
          ptrs[log.r] = n ## _malloc(log.a0);                   \
          break;                                                \
        case FREE:                                              \
          n ## _free(ptrs[log.a0]);                             \
          break;                                                \
        case CALLOC:                                            \
          ptrs[log.r] = n ## _calloc(log.a0, log.a1);           \
          break;                                                \
        case REALLOC:                                           \
          ptrs[log.r] = n ## _realloc(ptrs[log.a0], log.a1);    \
          break;                                                \
      }                                                         \
    }                                                           \
  }                                                             \

DEFINE_RUN_BENCH(__libc)
DEFINE_RUN_BENCH(kr)
DEFINE_RUN_BENCH(mmap)
DEFINE_RUN_BENCH(leak)
DEFINE_RUN_BENCH(my)

enum {
  KR,
  MMAP,
  GLIBC,
  LEAK,
  MY,
};

int main(int argc, char* argv[]) {
  int mode = KR;
  if (argc > 1) {
    const char* mode_str = argv[1];
    if (!strcmp(mode_str, "kr")) {
      mode = KR;
    } else if (!strcmp(mode_str, "mmap")) {
      mode = MMAP;
    } else if (!strcmp(mode_str, "glibc")) {
      mode = GLIBC;
    } else if (!strcmp(mode_str, "leak")) {
      mode = LEAK;
    } else if (!strcmp(mode_str, "my")) {
      mode = MY;
    } else {
      fprintf(stderr, "unknown mode\n");
      abort();
    }
  }

  ptrs.push_back(0);
  ptr2id[0] = 0;

  char buf[64];
  for (int i = 0; ~scanf("%s", buf); i++) {
    Log log = {};
    log.type = get_type(buf);
    void* p0;
    void* pr;
    switch (log.type) {
      case MALLOC:
        CHECK(scanf("%zu %p", &log.a0, &pr) == 2);
        record_alloc(pr, log.a0, &log);
        break;
      case FREE:
        CHECK(scanf("%p", &p0) == 1);
        record_free(p0, &log);
        break;
      case CALLOC:
        CHECK(scanf("%zu %zu %p", &log.a0, &log.a1, &pr) == 3);
        record_alloc(pr, log.a0 * log.a1, &log);
        break;
      case REALLOC:
        CHECK(scanf("%p %zu %p", &p0, &log.a1, &pr) == 3);
        record_free(p0, &log);
        record_alloc(pr, log.a1, &log);
        break;
      default:
        assert(0);
    }
    logs.push_back(log);
  }
  fprintf(stderr, "%zu bytes allocated by %zu mallocs, %zu unfreed\n",
          malloc_total, malloc_cnt, ptr2id.size() - 1);

  fprintf(stderr, "benchmark start!\n");
  clock_t start = clock();
  switch (mode) {
    case KR: run_bench_kr(); break;
    case MMAP: run_bench_mmap(); break;
    case GLIBC: run_bench___libc(); break;
    case LEAK: run_bench_leak(); break;
    case MY: run_bench_my(); break;
    default: abort();
  }
  double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("%f\n", elapsed);
}
