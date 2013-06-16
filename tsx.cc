#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if __GNUC__ * 100 + __GNUC_MINOR__ >= 407
# define USE_INTRIN
#endif

#if defined(USE_INTRIN)
# include <immintrin.h>
#endif

#include <chrono>
#include <iostream>
#include <random>
#include <string>

using namespace std;

static const int NUM_THREAD = 8;
static const int COUNT = 100000000;
extern "C" {
  unsigned int sum = 0;
}

struct SumParams {
  int freq;
  int seed;
};

template <class CommitPolicy>
void* sumRand(void* p) {
  SumParams* params = (SumParams*)p;
  int freq = params->freq;
  int seed = params->seed;

  assert(COUNT % freq == 0);
  unsigned int s = 0;
  const int times = COUNT / freq;
  mt19937 rg(seed);
  for (int t = 0; t < times; t++) {
    for (int i = 0; i < freq; i++) {
      s += rg();
    }
    CommitPolicy::commit(s);
    s = 0;
  }

  return NULL;
}

struct NoLockCommitPolicy {
  static void commit(unsigned int s) {
    sum += s;
  }
};

pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
struct MutexCommitPolicy {
  static void commit(int s) {
    pthread_mutex_lock(&mu);
    sum += s;
    pthread_mutex_unlock(&mu);
  }
};

struct AtomicCommitPolicy {
  static void commit(unsigned int s) {
    __sync_add_and_fetch(&sum, s);
  }
};

extern "C" {
  int tsx_cnt = 0;
}
struct TSXCommitPolicy {
  static void commit(unsigned int s) {
    asm volatile(".loop:\n"
                 " mov $1, %%eax;\n"
                 " add %%eax, tsx_cnt;\n"
                 " .byte 0xc7, 0xf8;\n"
                 " .long .fail-.begin;\n"
                 ".begin:\n"
                 " addl %0, sum;\n"
                 " .byte 0x0f, 0x01, 0xd5\n"
                 " jmp .done;\n"
                 ".fail:\n"
                 " jmp .loop;\n"
                 ".done:\n"
                 ::"r"(s):"%rax");
  }
};

#if defined(USE_INTRIN)
struct TSXIntrinCommitPolicy {
  static void commit(unsigned int s) {
    retry:
    int st = _xbegin();
    if (st == _XBEGIN_STARTED) {
      sum += s;
      _xend();
    } else {
      //printf("%x\n", st);
      goto retry;
    }
  }
};
#endif

struct GCCTransactionCommitPolicy {
  static void commit(unsigned int s) {
    __transaction_atomic {
      sum += s;
    }
  }
};

double getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

template <class CommitPolicy>
void run(string name, int freq) {
  sum = 0;
  double start = getTime();

  pthread_t th[NUM_THREAD];
  SumParams* params[NUM_THREAD];
  for (int i = 0; i < NUM_THREAD; i++) {
    SumParams* p = new SumParams();
    p->freq = freq;
    p->seed = i;
    params[i] = p;

    pthread_create(&th[i], NULL, &sumRand<CommitPolicy>, p);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    pthread_join(th[i], NULL);
    delete params[i];
  }

  cout << name << freq << ": " << sum << " "
       << (getTime() - start) << "s" << endl;
}

int main() {
  asm volatile(" .byte 0xc7, 0xf8;\n"
               " .long 3;\n"
               " .byte 0x0f, 0x01, 0xd5;\n");

  run<NoLockCommitPolicy>("nolock", 100);
  //run<NoLockCommitPolicy>("nolock", 100);
  run<MutexCommitPolicy>("mutex", 100);
  run<MutexCommitPolicy>("mutex", 10000);
  run<AtomicCommitPolicy>("atomic", 100);
  run<TSXCommitPolicy>("TSX", 100000000);
  printf("tsx_cnt=%d\n", tsx_cnt);
  tsx_cnt = 0;
  //sleep(1);
  run<TSXCommitPolicy>("TSX", 100);
  printf("tsx_cnt=%d\n", tsx_cnt);
  tsx_cnt = 0;
#if defined(USE_INTRIN)
  run<TSXIntrinCommitPolicy>("TSXIntrin", 100);
  printf("tsx_cnt=%d\n", tsx_cnt);
  tsx_cnt = 0;
#endif
  run<GCCTransactionCommitPolicy>("transaction", 100);
  run<NoLockCommitPolicy>("nolock", 100);
}
