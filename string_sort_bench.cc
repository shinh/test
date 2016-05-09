// License: GPLv3
// Data: http://shinh.skr.jp/dat_dir/sort.dat.xz
// Some code from https://github.com/bingmann/parallel-string-sorting/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <string>
#include <utility>
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

void CxxStableSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  for (auto& d : data) {
    stable_sort(d.begin(), d.end());
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


namespace ng_lcpmergesort {

/*
  Instead of just attaching the LLCPs to the strings to reduce memory usage,
  this code allocates n * ((size of key pointer) + (size of integer)) bytes of
  memory to store the annotated strings for clarity and simplicity.
*/

typedef unsigned int UINT;

typedef struct _AS {
    UINT llcp; char* str;
} AS, *PAS, *ASS; // annotated string sequence

void lcpm(PAS pX, PAS pXE, PAS pY, PAS pYE, ASS pZ)
{
    while ((pX<pXE) && (pY<pYE)) {
        if (*((UINT*)pX)!=*((UINT*)pY)) {
            if (*((UINT*)pX)>*((UINT*)pY)) {
                *pZ=*pX; pX++; pZ++;
            }
            else {
                *pZ=*pY; pY++; pZ++;
            }
        }
        else {
            char *x=pX->str+*((UINT*)pX);
            char *y=pY->str+*((UINT*)pX);
            while ((*x==*y) && (*x!=0)) { x++; y++; }
            if (*x>*y) {
                *(UINT*)pX=(x-(pX->str));
                *pZ=*pY; pY++; pZ++;
            }
            else {
                *(UINT*)pY=(y-(pY->str));
                *pZ=*pX; pX++; pZ++;
            }
        }
    }
    if (pX<pXE) {
        memcpy((char*)pZ, (char*)pX,
               (char*)pXE-(char*)pX);
    }
}

// for the last merge
void lcpm_f(PAS pX, PAS pXE, PAS pY, PAS pYE, char** pZ)
{
    while ((pX<pXE) && (pY<pYE)) {
        if (*(UINT*)pX!= *(UINT*)pY) {
            if (*(UINT*)pX > *(UINT*)pY) {

                *pZ=pX->str; pX++; pZ++;
            }
            else {
                *pZ=pY->str; pY++; pZ++;
            }
        }
        else {
            char *x=pX->str+*(UINT*)pX;
            char *y=pY->str+*(UINT*)pX;
            while ((*x==*y) && (*x!=0)) { x++; y++; }
            if (*x>*y) {
                *(UINT*)pX=x-pX->str;
                *pZ=pY->str; pY++; pZ++;
            }
            else {
                *(UINT*)pY=y-pY->str;
                *pZ=pX->str; pX++; pZ++;
            }
        }
    }
    while (pX<pXE) { *pZ=pX->str; pZ++; pX++; }
    while (pY<pYE) { *pZ=pY->str; pZ++; pY++; }
}

void lcpms_i(ASS pX, UINT n, char** str, ASS pZ)
{
    if (n>1) {
        UINT l=n/2;
        lcpms_i(pX+0, l-0, str+0, pZ);
        lcpms_i(pX+l, n-l, str+l, pZ);
        memcpy(pZ, pX, l*sizeof(AS));
        lcpm(pZ+0, pZ+l, pX+l, pX+n, pX);
    }
    else { pX->llcp=0; pX->str=*str; }
}

void lcpms(char** pStr, UINT n)
{
    ASS pX, pT;
    pX=pT=(ASS)malloc(sizeof(AS)*n);
    UINT l=n/2;
    lcpms_i(pX+0, l-0, pStr+0, (ASS)pStr);
    lcpms_i(pX+l, n-l, pStr+l, (ASS)pStr);
    lcpm_f(pX+0, pX+l, pX+l, pX+n, pStr);
    free(pT);
}

} // namespace ng_lcpmergesort


namespace inssort {

typedef unsigned char* string;

/******************************************************************************/

static inline void
inssort(string* str, int n, int d)
{
    string *pj, s, t;

    for (string* pi = str + 1; --n > 0; pi++) {
        string tmp = *pi;

        for (pj = pi; pj > str; pj--) {
            for (s = *(pj-1)+d, t = tmp+d; *s == *t && *s != 0; ++s, ++t)
                ;
            if (*s <= *t)
                break;
            *pj = *(pj-1);
        }
        *pj = tmp;
    }
}

static inline
void insertion_sort(string* a, size_t n)
{ inssort(a, n, 0); }

/******************************************************************************/

static inline void
inssort_range(string* str_begin, string* str_end, size_t depth)
{
    for (string* i = str_begin + 1; i != str_end; ++i) {
        string* j = i;
        string tmp = *i;
        while (j > str_begin) {
            string s = *(j - 1) + depth;
            string t = tmp + depth;
            while (*s == *t && *s != 0) ++s, ++t;
            if (*s <= *t) break;
            *j = *(j - 1);
            --j;
        }
        *j = tmp;
    }
}

/******************************************************************************/

//! Generic insertion sort for objectified string sets
template <typename StringSet>
static inline void inssort_generic(const StringSet& ss, size_t depth)
{
    typedef typename StringSet::Iterator Iterator;
    typedef typename StringSet::String String;
    typedef typename StringSet::CharIterator CharIterator;

    for (Iterator pi = ss.begin() + 1; pi != ss.end(); ++pi)
    {
        String tmp = std::move(ss[pi]);
        Iterator pj = pi;

        while (pj != ss.begin())
        {
            --pj;

            CharIterator s = ss.get_chars(ss[pj], depth);
            CharIterator t = ss.get_chars(tmp, depth);

            while (ss.is_equal(ss[pj], s, tmp, t))
                ++s, ++t;

            if (ss.is_leq(ss[pj], s, tmp, t)) {
                ++pj;
                break;
            }

            ss[pj + 1] = std::move(ss[pj]);
        }

        ss[pj] = std::move(tmp);
    }
}

/******************************************************************************/

} // namespace  inssort

namespace bs_mkqs {

typedef unsigned char* string;

#define i2c(i) x[i][depth]

static void vecswap(int i, int j, int n, string x[])
{
    while (n-- > 0) {
        std::swap(x[i], x[j]);
        i++;
        j++;
    }
}

static inline void ssort1(string x[], int n, int depth)
{
    int    a, b, c, d, r, v;
    if (n <= 1)
        return;
    a = rand() % n;
    std::swap(x[0], x[a]);
    v = i2c(0);
    a = b = 1;
    c = d = n-1;
    for (;;) {
        while (b <= c && (r = i2c(b)-v) <= 0) {
            if (r == 0) { std::swap(x[a], x[b]); a++; }
            b++;
        }
        while (b <= c && (r = i2c(c)-v) >= 0) {
            if (r == 0) { std::swap(x[c], x[d]); d--; }
            c--;
        }
        if (b > c) break;
        std::swap(x[b], x[c]);
        b++;
        c--;
    }
    r = std::min(a, b-a);     vecswap(0, b-r, r, x);
    r = std::min(d-c, n-d-1); vecswap(b, n-r, r, x);
    r = b-a; ssort1(x, r, depth);
    if (i2c(r) != 0)
        ssort1(x + r, a + n-d-1, depth+1);
    r = d-c; ssort1(x + n-r, r, depth);
}

static inline void multikey1(string x[], int n)
{ ssort1(x, n, 0); }


/* ssort2 -- Faster Version of Multikey Quicksort */

static inline void vecswap2(string *a, string *b, int n)
{
    while (n-- > 0) {
        string t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

#define ptr2char(i) (*(*(i) + depth))

static inline string *med3func(string *a, string *b, string *c, int depth)
{
    int va, vb, vc;
    if ((va=ptr2char(a)) == (vb=ptr2char(b)))
        return a;
    if ((vc=ptr2char(c)) == va || vc == vb)
        return c;
    return va < vb
        ? (vb < vc ? b : (va < vc ? c : a ) )
        : (vb > vc ? b : (va < vc ? a : c ) );
}

static inline void ssort2(string a[], size_t n, int depth)
{
    int d, r, partval;
    string *pa, *pb, *pc, *pd, *pl, *pm, *pn;
    if (n < 64) {
        return inssort::inssort(a, n, depth);
    }
    pl = a;
    pm = a + (n/2);
    pn = a + (n-1);
    if (n > 30) { /* On big arrays, pseudomedian of 9 */
        d = (n/8);
        pl = med3func(pl, pl+d, pl+2*d, depth);
        pm = med3func(pm-d, pm, pm+d,   depth);
        pn = med3func(pn-2*d, pn-d, pn, depth);
    }
    pm = med3func(pl, pm, pn, depth);
    std::swap(*a, *pm);
    partval = ptr2char(a);
    pa = pb = a + 1;
    pc = pd = a + n-1;
    for (;;) {
        while (pb <= pc && (r = ptr2char(pb)-partval) <= 0) {
            if (r == 0) std::swap(*pa++, *pb);
            pb++;
        }
        while (pb <= pc && (r = ptr2char(pc)-partval) >= 0) {
            if (r == 0) std::swap(*pc, *pd--);
            pc--;
        }
        if (pb > pc) break;
        std::swap(*pb++, *pc--);
    }
    pn = a + n;
    r = std::min(pa-a, pb-pa);    vecswap2(a,  pb-r, r);
    r = std::min(pd-pc, pn-pd-1); vecswap2(pb, pn-r, r);
    if ((r = pb-pa) > 1)
        ssort2(a, r, depth);
    if (ptr2char(a + r) != 0)
        ssort2(a + r, pa-a + pn-pd-1, depth+1);
    if ((r = pd-pc) > 1)
        ssort2(a + n-r, r, depth);
}

static inline void multikey2(string a[], size_t n)
{ ssort2(a, n, 0); }

static inline void bs_mkqsort(unsigned char **strings, size_t n)
{
    return multikey2(strings, n);
}

#undef i2c
#undef ptr2char

} // namespace bs_mkqs

void LcpMergeSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  vector<char*> ptrs;
  for (const auto& d : data) {
    ptrs.clear();
    for (const auto& s : d) {
      ptrs.push_back((char*)s.data());
    }
    ng_lcpmergesort::lcpms(&ptrs[0], ptrs.size());
  }
}

void BsMkqsSort(vector<vector<StringPiece>> data) {
  ScopedTimeReporter tr(__func__);
  vector<unsigned char*> ptrs;
  for (const auto& d : data) {
    ptrs.clear();
    for (const auto& s : d) {
      ptrs.push_back((unsigned char*)s.data());
    }
    bs_mkqs::bs_mkqsort(&ptrs[0], ptrs.size());
  }
}

struct AnnotatedString {
  const unsigned char* s;
  int l;
};

static inline pair<int, int> LcpCompare(const unsigned char* s1,
                                        const unsigned char* s2,
                                        int b) {
  int i;
  for (i = b;; i++) {
    unsigned char c1 = s1[i];
    unsigned char c2 = s2[i];
    if (c1 < c2)
      return make_pair(-1, i);
    else if (c1 > c2)
      return make_pair(1, i);
  }
  return make_pair(0, i);
}

void MyStringMergeSort(vector<const unsigned char*>& data,
                       int begin,
                       int end,
                       AnnotatedString* work,
                       AnnotatedString* out) {
#if 0
  if (begin + 1 == end) {
    out[begin] = AnnotatedString{ data[begin], 0 };
    return;
  }
#else
  if (begin + 64 >= end) {
    unsigned char** b = const_cast<unsigned char**>(&data[begin]);
    inssort::inssort(b, end - begin, 0);
    for (int i = begin; i < end; i++) {
      out[i] = AnnotatedString{ data[i], 0 };
    }
    return;
  }
#endif

  int mid = (begin + end) / 2;
  MyStringMergeSort(data, begin, mid, work, out);
  MyStringMergeSort(data, mid, end, work, out);

  memcpy(work, out + begin, (mid - begin) * sizeof(AnnotatedString));

  AnnotatedString* s1 = work;
  int s1_len = mid - begin;
  AnnotatedString* s2 = out + mid;
  int s2_len = end - mid;
  int i = 0, j = 0, k = 0;
  AnnotatedString* d = out + begin;
  for (; i < s1_len && j < s2_len; k++) {
    if (s1[i].l > s2[j].l) {
      d[k] = s1[i];
      i++;
    } else if (s1[i].l < s2[j].l) {
      d[k] = s2[j];
      j++;
    } else {
      pair<int, int> p = LcpCompare(s1[i].s, s2[j].s, s1[i].l);
      if (p.first < 0) {
        d[k] = s1[i];
        i++;
        s2[j].l = p.second;
      } else {
        d[k] = s2[j];
        j++;
        s1[i].l = p.second;
      }
    }
  }

  if (i < s1_len)
    memcpy(&d[k], &s1[i], (s1_len - i) * sizeof(AnnotatedString));
  else if (j < s2_len)
    memmove(&d[k], &s2[j], (s2_len - j) * sizeof(AnnotatedString));
}

void MyLcpMergeSort(vector<vector<const unsigned char*>> data) {
  vector<vector<StringPiece>> test;
  {
    ScopedTimeReporter tr(__func__);
    for (auto& d : data) {
      //vector<AnnotatedString> as(d.size());
      //vector<AnnotatedString> work(data.size() / 2 + 1);
      AnnotatedString* as = static_cast<AnnotatedString*>(
          malloc(d.size() * sizeof(AnnotatedString)));
      AnnotatedString* work = static_cast<AnnotatedString*>(
          malloc((d.size() / 2 + 1) * sizeof(AnnotatedString)));
      MyStringMergeSort(d, 0, d.size(), &work[0], &as[0]);
      free(as);
      free(work);
#if 0
      test.push_back(vector<StringPiece>());
      for (const AnnotatedString& s : as) {
        test.back().push_back(reinterpret_cast<const char*>(s.s));
      }
#endif
    }
  }

  for (size_t i = 0; i < test.size(); i++) {
    if (!is_sorted(test[i].begin(), test[i].end())) {
      for (const unsigned char* s : data[i]) {
      fprintf(stderr, "%s ", (const char*)s);
      }
      fprintf(stderr, "\n");
      for (StringPiece s : test[i]) {
      fprintf(stderr, "%.*s ", (int)s.size(), s.data());
      }
      fprintf(stderr, "\n");
      abort();
    }
  }
}

void MkQsort(unsigned char** strs, int n, int d) {
  if (n < 64) {
    inssort::inssort(strs, n, d);
    return;
  }

  char p1 = strs[0][d];
  char p2 = strs[n/2][d];
  char p3 = strs[n-1][d];
  if (p1 > p2)
    swap(p1, p2);
  if (p2 > p3) {
    swap(p2, p3);
    if (p1 > p2)
      swap(p1, p2);
  }

  const char p = p2;
  int j = 0;
  int k = n;
  for (int i = 0; i < k; i++) {
    const char c = strs[i][d];
    if (c < p) {
      swap(strs[i], strs[j]);
      j++;
    } else if (c > p) {
      swap(strs[i], strs[k-1]);
      k--;
      i--;
    }
  }

  MkQsort(strs, j, d);
  MkQsort(strs + j, k - j, d + 1);
  MkQsort(strs + k, n - k, d);
}

void MyMkQsort(vector<vector<const unsigned char*>> data) {
  {
    ScopedTimeReporter tr(__func__);
    for (auto& d : data) {
      MkQsort(const_cast<unsigned char**>(&d[0]), d.size(), 0);
    }
  }
}

int main() {
  vector<vector<StringPiece>> data;
  vector<vector<const unsigned char*>> ptrs;
  LoadData(&data);

  for (const auto& d : data) {
    vector<const unsigned char*> p;
    for (const auto& s : d) {
      p.push_back(reinterpret_cast<const unsigned char*>(s.data()));
    }
    ptrs.push_back(p);
  }

  fprintf(stderr, "Loaded %zu cases\n", data.size());

  //CxxSort(data);

  //StableListBasedRadixSort(data);
  MyMkQsort(ptrs);
  MyLcpMergeSort(ptrs);
  BsMkqsSort(data);
  LcpMergeSort(data);
  LibcSort(data);
  CxxSort(data);
  CxxStableSort(data);
}
