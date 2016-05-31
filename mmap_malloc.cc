#include <assert.h>
#include <string.h>
#include <sys/mman.h>

namespace {

void* malloc_impl(size_t size) {
  size = (size + sizeof(size_t) + 4095) & ~4095;
  size_t* r = static_cast<size_t*>(
      mmap(0, size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
  assert(r != MAP_FAILED);
  r[0] = size;
  return r + 1;
}

void* free_impl(void* ptr) {
  size_t* p = static_cast<size_t*>(ptr);
  p--;
  int r = munmap(p, *p);
  assert(r == 0);
}

}

extern "C" {

void* malloc(size_t size) {
  malloc_impl(size);
}

void free(void *ptr) {
  free_impl(ptr);
}

void* calloc(size_t nmemb, size_t size) {
  size *= nmemb;
  void* r = malloc(size);
  memset(r, 0, size);
  return r;
}

void* realloc(void *ptr, size_t size) {
  void* r = malloc(size);
  memcpy(r, ptr, size);
  free(ptr);
  return r;
}

}
