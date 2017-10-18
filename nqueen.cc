#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <queue>
#include <stack>
#include <vector>
#include <utility>

using namespace std;

struct State {
  State() : used_rows_(0) {
  }

  bool can_put(int x) const {
    if (used_rows_ & (1ULL << x))
      return false;
    int y = num_queens();
    for (int qy = 0; qy < num_queens(); qy++) {
      int qx = queens_[qy];
      if (abs(qx - x) == abs(qy - y))
        return false;
    }
    return true;
  }

  void put(int x) {
    used_rows_ |= 1ULL << x;
    queens_.push_back(x);
  }

  void revert(int x) {
    used_rows_ &= ~(1ULL << x);
    assert(queens_.back() == x);
    queens_.pop_back();
  }

  int num_queens() const {
    return static_cast<int>(queens_.size());
  }

  void show() const {
    for (int y = 0; y < N; y++) {
      int i = 0;
      if (y < num_queens()) {
        int x = queens_[y];
        for (; i < x; i++)
          printf(".");
        printf("Q");
        i++;
      }
      for (; i < N; i++)
        printf(".");
      puts("");

    }
    verify();
  }

  void verify() const {
    assert(num_queens() == N);
    vector<vector<char>> occupied(N);
    for (int y = 0; y < N; y++)
      occupied[y].resize(N);
    bool ok = true;
    for (int y = 0; y < N; y++) {
      int x = queens_[y];
      assert(x >= 0);
      assert(x < N);
      assert(y >= 0);
      assert(y < N);
      if (occupied[y][x]) {
        fprintf(stderr, "wrong: %d %d\n", x, y);
        ok = false;
      }
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0)
            continue;
          int nx = x + dx;
          int ny = y + dy;
          while (nx >= 0 && nx < N && ny >= 0 && ny < N) {
            occupied[ny][nx] = 1;
            nx += dx;
            ny += dy;
          }
        }
      }
    }
    assert(ok);
  }

  int prev() const { return queens_.back(); }

  static int N;

 protected:
  uint64_t used_rows_;
  vector<int> queens_;
};

int State::N;

struct ScoredState : public State {
  ScoredState() : State(), score_(0) {}

  void put(int x) {
    State::put(x);
    update_score();
  }

  void update_score() {
    int y = num_queens();
    score_ = y * 64 * 64;
    vector<char> cannot_put(N * N);
    for (int qy = 0; qy < N; qy++) {
      int qx = queens_[qy];
      for (int dx = -1; dx <= 1; dx += 2) {
        int x = (y - qy) * dx;
        int o = y * N;
        while (x >= 0 && x < N && o < N * N) {
          cannot_put[o + x] = 1;
          x += dx;
          o += N;
        }
      }
    }
    for (char c : cannot_put)
      score_ += !c;
  }

  int score() const { return score_; }

 private:
  int score_;
};

struct ScoredStateComparator {
  bool operator()(const ScoredState* a, const ScoredState* b) {
    return a->score() < b->score();
  }
};

namespace dfs {

void nqueen_rec(State* st, int n, int y) {
  if (n == y) {
    st->show();
    exit(0);
  }

  for (int x = 0; x < n; x++) {
    if (st->can_put(x)) {
      st->put(x);
      nqueen_rec(st, n, y + 1);
      st->revert(x);
    }
  }
}

void nqueen(int n) {
  State st;
  nqueen_rec(&st, n, 0);
}

}

namespace hdfs {

void nqueen_rec(State* st, int n, int y) {
  if (n == y) {
    st->show();
    exit(0);
  }

  auto go = [&](int x) {
    if (st->can_put(x)) {
      st->put(x);
      nqueen_rec(st, n, y + 1);
      st->revert(x);
    }
  };

  if (st->num_queens()) {
    int px = st->prev();
    if (px - 3 >= 0) {
      go(px - 3);
    }
    if (px + 3 < n) {
      go(px + 3);
    }
    for (int x = 0; x < n; x++) {
      if (x != px + 3 && x != px - 3) {
        go(x);
      }
    }
  } else {
    for (int x = 0; x < n; x++) {
      go(x);
    }
  }
}

void nqueen(int n) {
  State st;
  nqueen_rec(&st, n, 0);
}

}

namespace loop {

template <class Container>
typename Container::value_type pop(Container* c) {
  typename Container::value_type s = c->top();
  c->pop();
  return s;
}

// std::queue doesn't have top().
template <>
State* pop<>(queue<State*>* c) {
  State* s = c->front();
  c->pop();
  return s;
}

template <class Container, class S = State>
void nqueen(int n) {
  typedef S State;
  Container q;

  q.push(new State());
  while (!q.empty()) {
    State* st = pop(&q);

    int y = st->num_queens();
    for (int x = 0; x < n; x++) {
      if (st->can_put(x)) {
        State* nst = new State(*st);
        nst->put(x);
        if (y + 1 == n) {
          nst->show();
          exit(0);
        }
        q.push(nst);
      }
    }
    delete st;
  }

  fprintf(stderr, "Not found (should not happen)\n");
  exit(1);
}

}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: nqueen [dfs|bfs|ldfs|pq] <num>\n");
    return 1;
  }
  int n = static_cast<int>(strtol(argv[2], nullptr, 10));
  if (n > 64) {
    fprintf(stderr, "Cannot specify num>64\n");
    return 1;
  }
  State::N = n;
  if (!strcmp(argv[1], "dfs")) {
    dfs::nqueen(n);
  } else if (!strcmp(argv[1], "bfs")) {
    loop::nqueen<queue<State*>>(n);
  } else if (!strcmp(argv[1], "ldfs")) {
    loop::nqueen<stack<State*>>(n);
  } else if (!strcmp(argv[1], "pq")) {
    loop::nqueen<priority_queue<ScoredState*, vector<ScoredState*>,
                                ScoredStateComparator>,
                 ScoredState>(n);
  } else if (!strcmp(argv[1], "hdfs")) {
    hdfs::nqueen(n);
  } else {
    fprintf(stderr, "Unknown mode: %s\n", argv[1]);
  }
}
