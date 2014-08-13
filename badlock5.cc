#include <pthread.h>

class C {
public:
  C() : val_(3) {
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlockattr_setkind_np(
        &attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    pthread_rwlock_init(&mu_, &attr);
    pthread_rwlockattr_destroy(&attr);
  }
  int calcSomething() {
    pthread_rwlock_rdlock(&mu_);
    int r = 42 + getVal();
    pthread_rwlock_unlock(&mu_);
    return r;
  }
  int getVal() {
    pthread_rwlock_rdlock(&mu_);
    int r = val_;
    pthread_rwlock_unlock(&mu_);
    return r;
  }
  void setVal(int val) {
    pthread_rwlock_wrlock(&mu_);
    val_ = val;
    pthread_rwlock_unlock(&mu_);
  }
private:
  pthread_rwlock_t mu_;
  int val_;
};

void* thread(void* data) {
  C* c = (C*)data;
  for (int i = 0; i < 100000; i++) {
    c->calcSomething();
    c->setVal(i);
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
