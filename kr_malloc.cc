#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

namespace {

static const uint32_t CHUNK_SIZE = 32;

struct ChunkHeader {
  size_t sz;
  ChunkHeader* fd;
  ChunkHeader* bk;
};

static const uint32_t HEADER_SIZE = sizeof(size_t);

template <typename T> inline T add(T a, uintptr_t x) {
  uintptr_t v = reinterpret_cast<uintptr_t>(a);
  return reinterpret_cast<T>(v + x);
}

template <typename T> inline T sub(T a, uintptr_t x) {
  uintptr_t v = reinterpret_cast<uintptr_t>(a);
  return reinterpret_cast<T>(v - x);
}

template <typename T> inline T align(T a) {
  uintptr_t v = reinterpret_cast<uintptr_t>(a);
  v &= ~(CHUNK_SIZE - 1);
  return reinterpret_cast<T>(v);
}

template <typename T> inline T align_up(T a) {
  return align(add(a, CHUNK_SIZE - 1));
}

ChunkHeader* mem;
ChunkHeader* freep;

void init() {
  const size_t sz = 4ULL * 1024 * 1024 * 1024;
  mem = static_cast<ChunkHeader*>(
      mmap(0, sz, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
  assert(mem != MAP_FAILED);
  mem->sz = sz;
  freep = mem;
}

void* malloc_impl(size_t size) {
  if (!mem)
    init();

  // Find the first fit.
  size = align_up(size + HEADER_SIZE);
  while (size > freep->sz) {
    // TODO: loop
    assert(freep->fd);
    freep = freep->fd;
  }

  ChunkHeader* r = freep;
  ChunkHeader* bk = r->bk;
  ChunkHeader* fd = r->fd;
  if (size < r->sz) {
    // Split the current chunk.
    ChunkHeader* p = add(r, size);
    p->sz = r->sz - size;
    p->fd = fd;
    p->bk = bk;
    r->sz = size | 1U;

    if (fd) {
      fd->bk = p;
    }
    fd = p;
  } else {
    assert(fd);
  }

  fd->bk = bk;
  if (bk) {
    bk->fd = fd;
  }
  freep = fd;
  r->fd = fd;

  return add(r, HEADER_SIZE);
}

void free_impl(void* ptr) {
  if (!ptr)
    return;

  ChunkHeader* p = static_cast<ChunkHeader*>(sub(ptr, HEADER_SIZE));
  size_t sz = p->sz & ~1;

  //fprintf(stderr, "p=%p freep=%p %d\n", p, freep, p < freep);
  ChunkHeader* fd = freep;
  if (p < fd) {
    while (p < fd->bk) {
      fd = fd->bk;
    }
  } else {
    while (p > fd) {
      assert(fd->fd);
      fd = fd->fd;
    }
  }
  //fprintf(stderr, "p=%p freep=%p fd=%p\n", p, freep, fd);
  ChunkHeader* bk = fd->bk;

  // TODO: consolidate free chunks.
  p->sz = sz;
  p->bk = bk;
  p->fd = fd;
  if (bk) {
    bk->fd = p;
  } else {
    // TODO: for loop
  }
  fd->bk = p;

  freep = p;
}

}

extern "C" {

void* my_malloc(size_t size) {
  void* r = malloc_impl(size);
  //fprintf(stderr, "malloc %zu %p\n", size, r);
  return r;
}

void my_free(void *ptr) {
  //fprintf(stderr, "free %p\n", ptr);
  free_impl(ptr);
}

void* my_calloc(size_t nmemb, size_t size) {
  size *= nmemb;
  void* r = malloc(size);
  memset(r, 0, size);
  return r;
}

void* my_realloc(void *ptr, size_t size) {
  void* r = malloc(size);
  memcpy(r, ptr, size);
  free(ptr);
  return r;
}

}
