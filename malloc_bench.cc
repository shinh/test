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

extern "C" {
void* my_malloc(size_t size);
void my_free(void *ptr);
void* my_calloc(size_t nmemb, size_t size);
void* my_realloc(void *ptr, size_t size);
#define malloc my_malloc
#define free my_free
#define calloc my_calloc
#define realloc my_realloc
}

void run_bench() {
  for (const Log& log : logs) {
    switch (log.type) {
      case MALLOC:
        ptrs[log.r] = malloc(log.a0);
        break;
      case FREE:
        free(ptrs[log.a0]);
        break;
      case CALLOC:
        ptrs[log.r] = calloc(log.a0, log.a1);
        break;
      case REALLOC:
        ptrs[log.r] = realloc(ptrs[log.a0], log.a1);
        break;
    }
  }
}

int main() {
  ptrs.push_back(0);
  ptr2id[0] = 0;

  char buf[64];
  while (~scanf("%s", buf)) {
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
  run_bench();
  double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("%f\n", elapsed);
}
