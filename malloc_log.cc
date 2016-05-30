// g++ -fPIC -O -g -Wall -W malloc_log.cc -shared -o malloc_log.so

#include <stdio.h>

using namespace std;

extern "C" {

extern void* __libc_malloc(size_t size);
extern void __libc_free(void* ptr);
extern void* __libc_calloc(size_t nmemb, size_t size);
extern void* __libc_realloc(void* ptr, size_t size);

void* malloc(size_t size) {
  void* r = __libc_malloc(size);
  fprintf(stderr, "MLOG: malloc(%zu) => %p\n", size, r);
  return r;
}

void free(void* ptr) {
  __libc_free(ptr);
  fprintf(stderr, "MLOG: free(%p)\n", ptr);
}

void* calloc(size_t nmemb, size_t size) {
  void* r = __libc_calloc(nmemb, size);
  fprintf(stderr, "MLOG: calloc(%zu, %zu) => %p\n", nmemb, size, r);
  return r;
}

void* realloc(void* ptr, size_t size) {
  void* r = __libc_realloc(ptr, size);
  fprintf(stderr, "MLOG: realloc(%p, %zu) => %p\n", ptr, size, r);
  return r;
}

}
