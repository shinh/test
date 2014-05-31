#include <assert.h>
#include <pthread.h>
#include <vector>

class C {
public:
  explicit C() {
    pthread_mutex_init(&mu_, NULL);
  }
  void setVals(std::vector<int> vals) {
    pthread_mutex_lock(&mu_);
    vals_ = vals;
    pthread_mutex_unlock(&mu_);
    // Update the cache.
    setSum(calcSum());
  }
  int getSum() {
    pthread_mutex_lock(&mu_);
    int sum = sum_;
    pthread_mutex_unlock(&mu_);
    return sum;
  }
  void setSum(int sum) {
    pthread_mutex_lock(&mu_);
    sum_ = sum;
    pthread_mutex_unlock(&mu_);
  }
  void check() {
    pthread_mutex_lock(&mu_);
    int sum = 0;
    for (size_t i = 0; i < vals_.size(); i++)
      sum += vals_[i];
    assert(sum == sum_);
    pthread_mutex_unlock(&mu_);
  }
private:
  int calcSum() {
    pthread_mutex_lock(&mu_);
    int sum = 0;
    for (size_t i = 0; i < vals_.size(); i++)
      sum += vals_[i];
    pthread_mutex_unlock(&mu_);
    return sum;
  }
  pthread_mutex_t mu_;
  std::vector<int> vals_;
  int sum_;
};

void* thread(void* data) {
  C* c = (C*)data;
  std::vector<int> vals;
  for (int i = 0; i < 1000; i++) {
    vals.push_back(i);
    c->setVals(vals);
    c->check();
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
