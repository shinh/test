#include <pthread.h>

#define NTHREAD 2

pthread_mutex_t mu1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mu2 = PTHREAD_MUTEX_INITIALIZER;

void* thread_func(void* arg) {
  for (int i = 0; i < 100; i++) {
    pthread_mutex_lock(&mu1);
    pthread_mutex_lock(&mu2);
    pthread_mutex_unlock(&mu2);
    pthread_mutex_unlock(&mu1);

    pthread_mutex_lock(&mu2);
    pthread_mutex_lock(&mu1);
    pthread_mutex_unlock(&mu1);
    pthread_mutex_unlock(&mu2);
  }
}

int main() {
  pthread_t th[NTHREAD];
  for (int i = 0; i < NTHREAD; i++) {
    pthread_create(&th[i], NULL, &thread_func, NULL);
  }
  for (int i = 0; i < NTHREAD; i++) {
    pthread_join(th[i], NULL);
  }
}
