#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

class C {
public:
  C() : done_(false), sum_(0) {
    pthread_mutex_init(&mu_, NULL);
    pthread_cond_init(&cond_, NULL);
  }
  void wait() {
    pthread_mutex_lock(&mu_);
    pthread_cond_wait(&cond_, &mu_);
    pthread_mutex_unlock(&mu_);
  }
  void doSomething() {
    pthread_mutex_lock(&mu_);
    done_ = true;
    pthread_mutex_unlock(&mu_);
    pthread_cond_signal(&cond_);
  }
private:
  pthread_mutex_t mu_;
  pthread_cond_t cond_;
  bool done_;
  int sum_;
};

void* thread(void* data) {
  C* c = (C*)data;
  c->doSomething();
  return NULL;
}

int main() {
  C* c = new C();
  pthread_t th;
  pthread_create(&th, NULL, &thread, c);
  // usleep(1);
  c->wait();
  pthread_join(th, NULL);
  delete c;

}
