#include <assert.h>
#include <string.h>
#include <sys/mman.h>

#include "malloc_bench.h"

namespace {

char* mem;

void init() {
  const size_t sz = 4ULL * 1024 * 1024 * 1024;
  mem = static_cast<char*>(
      mmap(0, sz, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
  assert(mem != MAP_FAILED);
}

void* malloc_impl(size_t size) {
  if (!mem)
    init();
  void* r = mem;
  mem += size;
  return r;
}

void* free_impl(void* ptr) {
}

}

void* leak_malloc(size_t size) {
  malloc_impl(size);
}

void leak_free(void *ptr) {
  free_impl(ptr);
}

DEFINE_CALLOC(leak)
DEFINE_REALLOC(leak)
