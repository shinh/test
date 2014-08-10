#include <pthread.h>
#include <vector>

pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

class C {
public:
  C() {
    pthread_mutex_init(&mu_, NULL);
  }
  void doSlowOperation() {
    pthread_mutex_unlock(&g_mu);
    pthread_mutex_lock(&mu_);
    for (int i = 0; i < 5; i++)
      vals_.push_back(i);
    pthread_mutex_unlock(&mu_);
    pthread_mutex_lock(&g_mu);
  }
  void doFastOperation() {
    vals_.push_back(42);
  }
private:
  pthread_mutex_t mu_;
  std::vector<int> vals_;
};

void* thread(void* data) {
  C* c = (C*)data;
  for (int i = 0; i < 100000; i++) {
    pthread_mutex_lock(&g_mu);
    c->doSlowOperation();
    c->doFastOperation();
    pthread_mutex_unlock(&g_mu);
  }
  return NULL;
}

int main() {
  C* c = new C();
  pthread_t th1;
  pthread_create(&th1, NULL, &thread, c);
  pthread_t th2;
  pthread_create(&th2, NULL, &thread, c);
  pthread_join(th1, NULL);
  pthread_join(th2, NULL);
}
