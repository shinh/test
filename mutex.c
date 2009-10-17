#include <stdio.h>
#include <pthread.h>

#define USE_PTHREAD

long long cnt;

#ifdef USE_PTHREAD
pthread_mutex_t mu;
#endif

void* count_up(void* idp) {
    int id = (int)idp;
    int i;
    printf("thread %d start\n", id);
    for (i = 0; i < 10000000; i++) {
#ifdef USE_PTHREAD
        pthread_mutex_lock(&mu);
        cnt++;
        pthread_mutex_unlock(&mu);
#else
        cnt++;
#endif
    }
    printf("thread %d end\n", id);
    return NULL;
}

#define NUM_THREADS 10

int main() {
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
}
