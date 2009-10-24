#include <stdio.h>
#include <pthread.h>
#include <sched.h>

//#define USE_PTHREAD_MUTEX
//#define USE_PTHREAD_SPINLOCK
//#define USE_CMPXCHG
#define USE_CMPXCHG2
//#define USE_SYNC_ADD
//#define USE_PPC
//#define USE_PPC2

#if defined(USE_PTHREAD_MUTEX)
pthread_mutex_t mu;
#elif defined(USE_PTHREAD_SPINLOCK)
pthread_spinlock_t mu;
#else
int mu;
#endif

int cnt;

void* count_up(void* idp) {
    int id = (int)idp;
    int i;
    printf("thread %d start\n", id);
    for (i = 0; i < 10000000; i++) {
#if defined(USE_PTHREAD_MUTEX)
        pthread_mutex_lock(&mu);
        cnt++;
        pthread_mutex_unlock(&mu);
#elif defined(USE_PTHREAD_SPINLOCK)
        pthread_spin_lock(&mu);
        cnt++;
        pthread_spin_unlock(&mu);
#elif defined(USE_CMPXCHG)
        while (__sync_val_compare_and_swap(&mu, 0, 1)) {
            sched_yield();
        }
        cnt++;
        mu = 0;
#elif defined(USE_CMPXCHG2)
        while (1) {
            int v = cnt;
            if (__sync_val_compare_and_swap(&cnt, v, v+1) == v) break;
            sched_yield();
        }
#elif defined(USE_SYNC_ADD)
        __sync_add_and_fetch(&cnt, 1);
#elif defined(USE_PPC)
        int* mup;
        asm(" b .yielded;\n"
            ".yield:\n"
            " bl sched_yield;\n"
            ".yielded:\n");
        mup = &mu;
        asm(".lock_cnt:\n"
            " lwarx 7, 0, %0;\n"
            " cmpwi 7, 0;\n"
            " bne .yield;\n"
            " addi 7, 7, 1;\n"
            " stwcx. 7, 0, %0;\n"
            " bne- .lock_cnt;\n"
            :"=r"(mup)::"7");
        cnt++;
        mu = 0;
#elif defined(USE_PPC2)
        int* cntp = &cnt;
        asm(".lock_cnt:\n"
            " lwarx 7, 0, %0;\n"
            " addi 7, 7, 1;\n"
            " stwcx. 7, 0, %0;\n"
            " bne- .lock_cnt;\n"
            :"=r"(cntp)::"7");
#else
        cnt++;
#endif
    }
    printf("thread %d end\n", id);
    return NULL;
}

#define NUM_THREADS 10

int main() {
#if defined(USE_PTHREAD_MUTEX)
    pthread_mutex_init(&mu, NULL);
#elif defined(USE_PTHREAD_SPINLOCK)
    pthread_spin_init(&mu, PTHREAD_PROCESS_PRIVATE);
#endif

    int i;
    pthread_t th[NUM_THREADS];
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_create(&th[i], NULL, count_up, (void*)i);
    }

    void* dummy;
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(th[i], &dummy);
    }

    printf("%d\n", cnt);

#if defined(USE_PTHREAD_MUTEX)
    pthread_mutex_destroy(&mu);
#elif defined(USE_PTHREAD_SPINLOCK)
    pthread_spin_destroy(&mu);
#endif
}
