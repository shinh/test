#include <stdio.h>
#include <pthread.h>
#include <sched.h>

//#define USE_PTHREAD_MUTEX
//#define USE_PTHREAD_SPINLOCK
#define USE_CMPXCHG

long long cnt;

#if defined(USE_PTHREAD_MUTEX)
pthread_mutex_t mu;
#elif defined(USE_PTHREAD_SPINLOCK)
pthread_spinlock_t mu;
#elif defined(USE_CMPXCHG)
int mu;
#endif

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

    printf("%lld\n", cnt);

#if defined(USE_PTHREAD_MUTEX)
    pthread_mutex_destroy(&mu);
#elif defined(USE_PTHREAD_SPINLOCK)
    pthread_spin_destroy(&mu);
#endif
}
