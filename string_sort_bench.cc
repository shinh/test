// Data: http://shinh.skr.jp/dat_dir/sort.dat.xz

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <string>
#include <vector>

#include "string_piece.h"

using namespace std;

struct ScopedTimeReporter {
 public:
  explicit ScopedTimeReporter(const char* name);
  ~ScopedTimeReporter();

 private:
  const char* name_;
  double start_;
};

double GetTime() {
#if defined(__linux__)
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 0.001 * 0.001 * 0.001;
#else
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    perror("gettimeofday");
    exit(1);
  }
  return tv.tv_sec + tv.tv_usec * 0.001 * 0.001;
#endif
}

ScopedTimeReporter::ScopedTimeReporter(const char* name)
    : name_(name), start_(GetTime()) {
}

ScopedTimeReporter::~ScopedTimeReporter() {
  double elapsed = GetTime() - start_;
  fprintf(stderr, "%s: %f\n", name_, elapsed);
}

void LoadData(vector<vector<StringPiece>>* data) {
  FILE* fp = fopen("sort.dat", "rb");
  if (!fp) {
    fprintf(stderr,
            "Download data from http://shinh.skr.jp/dat_dir/sort.dat.xz\n");
    exit(1);
  }

  while (true) {
    char* buf = nullptr;
    size_t len;
    if (getline(&buf, &len, fp) < 0)
      return;
    if (*buf == 0)
      return;

    vector<StringPiece> toks;
    char* st = buf;
    for (char* p = st;; p++) {
      char c = *p;
      if (c == ' ' || c == '\n') {
        toks.push_back(StringPiece(st, p - st));
        *p = 0;
        if (c == '\n')
          break;
        st = p + 1;
      }
    }

    data->push_back(toks);
  }
}

void CxxSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  for (auto& d : data) {
    sort(d.begin(), d.end());
  }
}

static int CompareStringPiece(const void* v1, const void* v2) {
  auto s1 = reinterpret_cast<const StringPiece*>(v1);
  auto s2 = reinterpret_cast<const StringPiece*>(v2);
  return s1->compare(*s2);
}

void LibcSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  for (auto& d : data) {
    qsort(&d[0], d.size(), sizeof(d[0]), &CompareStringPiece);
  }
}

// https://www.usenix.org/legacy/publications/compsystems/1993/win_mcilroy.pdf

#define SIZE 10000

struct list {
  list* next;
  const unsigned char* data;
};

list* rsort(list* a) {
#define push(a, t, b) sp->sa = a, sp->st = t, (sp++)->sb = b
#define pop(a, t, b) a = (--sp)->sa, t = sp->st, b = sp->sb
#define stackempty() (sp <= stack)
#define singleton(a) (a->next == 0)
#define ended(a, b) (b > 0 && a->data[b-1] == 0)
  struct { list* sa; list* st; int sb; } stack[SIZE], *sp = stack;
  static list* pile[256];
  static list* tail[256];
  list* atail;
  list* sequel = 0;
  int b, c, cmin, nc = 0;

  if (a && !singleton(a))
    push(a, 0, 0);

  while (!stackempty()) {
    pop(a, atail, b);
    if (singleton(a) || ended(a, b)) {
      atail->next = sequel;
      sequel = a;
      continue;
    }

    cmin = 255;
    for (; a; a = a->next) {
      c = a->data[b];
      if (pile[c] == 0) {
        tail[c] = pile[c] = a;
        if (c == 0)
          continue;
        if (c < cmin)
          cmin = c;
        nc++;
      } else {
        // LBTM :(
        tail[c] = tail[c]->next = a;
      }
    }

    if (pile[0]) {
      push(pile[0], tail[0], b+1);
      tail[0]->next = pile[0] = 0;
    }

    for (c = cmin; nc > 0; c++) {
      if (pile[c]) {
        push(pile[c], tail[c], b+1);
        tail[c]->next = pile[0] = 0;
        nc--;
      }
    }

  }
  return sequel;
}

void StableListBasedRadixSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  for (auto& d : data) {
    vector<list> l(d.size());
    for (size_t i = 0; i < d.size(); i++) {
      l[i].next = (i == d.size() - 1) ? nullptr : &l[i+1];
      l[i].data = reinterpret_cast<const unsigned char*>(d[i].data());
    }
    rsort(&l[0]);
  }
}

int main() {
  vector<vector<StringPiece>> data;
  LoadData(&data);
  fprintf(stderr, "Loaded %zu cases\n", data.size());

  //CxxSort(data);

  StableListBasedRadixSort(data);
  LibcSort(data);
  CxxSort(data);
}
