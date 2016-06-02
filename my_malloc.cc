#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "malloc_bench.h"

namespace {

struct ChunkHeader {
  size_t sz;
  ChunkHeader* fd;
  ChunkHeader* bk;
};

static const uint32_t CHUNK_SIZE_BITS = 5;
static const uint32_t CHUNK_SIZE = 1 << CHUNK_SIZE_BITS;
static const uint32_t HEADER_SIZE = sizeof(size_t);
static const uint32_t MMAP_THRESHOLD = 256 * 1024;
static const uint32_t FASTBIN_THRESHOLD_BITS = 12;
static const uint32_t FASTBIN_THRESHOLD = 1 << FASTBIN_THRESHOLD_BITS;

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
ChunkHeader* fastbin[32];

void init() {
  const size_t sz = 4ULL * 1024 * 1024 * 1024;
  mem = static_cast<ChunkHeader*>(
      mmap(0, sz, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0));
  assert(mem != MAP_FAILED);
  mem->sz = sz;
  freep = mem;
}

int get_bin_index(size_t size) {
  int bi = 32 - __builtin_clz(size);
  assert(bi > CHUNK_SIZE_BITS);
  if (bi < FASTBIN_THRESHOLD_BITS)
    return bi - 1;
  return 0;
}

void* malloc_impl(size_t size) {
  if (!mem)
    init();

  if (size == 0)
    return 0;

  size = align_up(size + HEADER_SIZE);
  if (size >= MMAP_THRESHOLD) {
    return mmap_malloc(size);
  }

  int fbi = get_bin_index(size);
  if (fbi) {
    size = 1 << fbi;
    if (fastbin[fbi]) {
      ChunkHeader* r = fastbin[fbi];
      fastbin[fbi] = r->fd;
      assert(r->sz == size);
      return add(r, HEADER_SIZE);
    }
  }

  // Find the first fit.
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
  assert(sz);
  if (sz >= MMAP_THRESHOLD) {
    mmap_free(ptr);
    return;
  }

  int fbi = get_bin_index(sz);
  if (fbi) {
    p->sz = sz;
    p->fd = fastbin[fbi];
    fastbin[fbi] = p;
    return;
  }

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

void* my_malloc(size_t size) {
  void* r = malloc_impl(size);
  //fprintf(stderr, "malloc %zu %p\n", size, r);
  return r;
}

void my_free(void *ptr) {
  //fprintf(stderr, "free %p\n", ptr);
  free_impl(ptr);
}

DEFINE_CALLOC(my)
DEFINE_REALLOC(my)
