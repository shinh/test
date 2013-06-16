#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <atomic>

using namespace std;

#define BUF_SIZE (1<<16)
char buf[BUF_SIZE] __attribute__((aligned(1<<16)));

void xabort_explict() {
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    _xabort(42);
  } else {
    printf("%s: %x %d\n", __func__, (st & ((1 << 6) - 1)), _XABORT_CODE(st));
  }
}

void xabort_capacity_sleep() {
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    sleep(1);
    _xend();
  } else {
    printf("%s: %x\n", __func__, st);
  }
}

void xabort_capacity_loop() {
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    for (;;) {}
    _xend();
  } else {
    printf("%s: %x\n", __func__, st);
  }
}

void xabort_capacity_copy(int s) {
  buf[s] = 0;
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    for (int i = 0; i < s; i++)
      buf[i] = i;
    buf[s] = 42;
    _xend();
  } else {
    printf("%s%d: %x %d\n", __func__, s, st, buf[s]);
  }
}

void xabort_twice() {
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    int st2 = _xbegin();
    if (st2 == _XBEGIN_STARTED) {
      _xend();
    } else {
      printf("%s2: %x\n", __func__, st2);
    }
    _xend();
  } else {
    printf("%s1: %x\n", __func__, st);
  }
}

void xabort_recursive(int d, int m) {
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    if (m <= 0 || d < m)
      xabort_recursive(d + 1, m);
    _xend();
  } else {
    printf("%s%d: %x %d\n", __func__, m, st, d);
  }
}

atomic<int> started;

void* xabort_conflict_thread(void* d) {
  int gap = *(int*)d;
  started.store(1);
  for (int i = 0; i < 200; i++)
    __sync_add_and_fetch(&buf[gap], i);
}

void xabort_conflict(int gap) {
  pthread_t th;
  started.store(0);
  pthread_create(&th, NULL, xabort_conflict_thread, &gap);
  while (started.load() == 0) {}
  unsigned st = _xbegin();
  if (st == _XBEGIN_STARTED) {
    for (int i = 0; i < 200; i++)
      __sync_add_and_fetch(&buf[0], i);
    _xend();
  } else {
    printf("%s%d: %x\n", __func__, gap, st);
  }
  pthread_join(th, NULL);
}

int main() {
  xabort_explict();
  xabort_capacity_sleep();
  xabort_capacity_loop();
  xabort_capacity_copy(1<<8);
  xabort_capacity_copy(1<<9);
  xabort_capacity_copy(1<<10);
  xabort_capacity_copy(1<<11);
  xabort_capacity_copy(1<<12);
  xabort_capacity_copy(1<<15);
  xabort_twice();
  xabort_recursive(0, 0);
  xabort_recursive(0, 2);
  xabort_recursive(0, 5);
  xabort_recursive(0, 6);
  xabort_recursive(0, 7);
  xabort_recursive(0, 8);
  xabort_conflict(0);
  xabort_conflict(30);
  xabort_conflict(40);
  xabort_conflict(63);
  xabort_conflict(64);
  xabort_conflict(70);
  xabort_conflict(4096);
}
