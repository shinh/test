class NullList {};

template<typename H, typename T>
struct TypeList {
  typedef H Head;
  typedef T Tail;
};

#define TLIST0() NullList
#define TLIST1(a) TypeList<a, TLIST0() >
#define TLIST2(a, b) TypeList<a, TLIST1(b) >
#define TLIST3(a, b, c) TypeList<a, TLIST2(b, c) >
#define TLIST4(a, b, c, d) TypeList<a, TLIST3(b, c, d) >

TLIST3(char, short, long) tl;
