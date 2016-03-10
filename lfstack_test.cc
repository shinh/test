#include <assert.h>

#include <algorithm>
#include <vector>

#include "lfstack.h"
#include "log.h"
#include "testutil.h"
#include "thread.h"

namespace {

void TestSingleThread() {
  LockFreeStack<int> st;
  ASSERT_EQ(st.empty(), true);
  st.push(42);
  ASSERT_EQ(st.empty(), false);
  st.push(43);
  st.push(44);
  ASSERT_EQ(st.empty(), false);
  ASSERT_EQ(st.pop(), 44);
  ASSERT_EQ(st.empty(), false);
  ASSERT_EQ(st.pop(), 43);
  ASSERT_EQ(st.empty(), false);
  ASSERT_EQ(st.pop(), 42);
  ASSERT_EQ(st.empty(), true);
}

void TestMultiThread() {
  static const int kNumThreads = 100;
  static const int kNumEntries = 10000;
  LockFreeStack<int> st;
  vector<thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(thread([i, &st]() {
          for (int j = 0; j < kNumEntries; j++) {
            st.push(i * kNumEntries + j);
          }
        }));
  }

  for (int i = 0; i < kNumThreads; i++) {
    threads[i].join();
  }
  threads.clear();

  threads.reserve(kNumThreads);
  vector<vector<int>> results(kNumThreads);
  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(thread([i, &st, &results]() {
          for (int j = 0; j < kNumEntries; j++) {
            results[i].push_back(st.pop());
          }
        }));
  }

  for (int i = 0; i < kNumThreads; i++) {
    threads[i].join();
  }
  threads.clear();

  vector<int> sorted;
  for (int i = 0; i < kNumThreads; i++) {
    for (int v : results[i]) {
      sorted.push_back(v);
    }
  }
  sort(sorted.begin(), sorted.end());

  for (int i = 0; i < kNumThreads * kNumEntries; i++) {
    ASSERT_EQ(sorted[i], i);
  }
}

}  // namespace

int main() {
  g_log_no_exit = true;
  TestSingleThread();
  TestMultiThread();
  assert(!g_failed);
}
