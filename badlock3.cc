#include <assert.h>
#include <pthread.h>

class C {
public:
  explicit C(int val) : val_(val) {
    pthread_mutex_init(&mu_, NULL);
  }
  int add(C* c) {
    pthread_mutex_lock(&mu_);
    pthread_mutex_lock(&c->mu_);
    int r = val_ + c->val_;
    pthread_mutex_unlock(&c->mu_);
    pthread_mutex_unlock(&mu_);
    return r;
  }
private:
  pthread_mutex_t mu_;
  int val_;
};

void* thread(void* data) {
  C** cs = (C**)data;
  for (int i = 0; i < 100000; i++) {
    int v = cs[0]->add(cs[1]);
    assert(v == 42 + 43);
  }
  return NULL;
}

int main() {
  C* c1 = new C(42);
  C* c2 = new C(43);
  C* cs1[2] = { c1, c2 };
  C* cs2[2] = { c2, c1 };
  pthread_t th1;
  pthread_create(&th1, NULL, &thread, cs1);
  pthread_t th2;
  pthread_create(&th2, NULL, &thread, cs2);
  pthread_join(th1, NULL);
  pthread_join(th2, NULL);
}
