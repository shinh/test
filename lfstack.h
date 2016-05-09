#ifndef LFSTACK_H_
#define LFSTACK_H_

#include <assert.h>

#include <atomic>

using namespace std;

template <class T>
class LockFreeStack {
  struct Node {
    T v;
    Node* next;
  };

 public:
  LockFreeStack() : top_(nullptr) {}

  void push(T v) {
    Node* n = new Node;
    n->v = v;
    n->next = top_.load();
    while (!top_.compare_exchange_weak(n->next, n)) {}
  }

  T pop() {
    Node* n = top_.load();
    while (!top_.compare_exchange_weak(n, n->next)) {}
    T r = n->v;
    delete n;
    return r;
  }

  bool empty() const {
    return top_ == nullptr;
  }

 private:
  atomic<Node*> top_;
};

#endif
