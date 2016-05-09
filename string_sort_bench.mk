STRING_SORT_BENCH_ALL := string_sort_bench
STRING_SORT_BENCH_OBJS := string_piece.o string_sort_bench.o
STRING_SORT_BENCH_CXXFLAGS := -g -W -Wall -O2 -std=c++11 -MMD -MP
STRING_SORT_BENCH_CXX := $(CXX)

STRING_SORT_BENCH_CXX := clang++-3.6
STRING_SORT_BENCH_CXXFLAGS += -nostdinc++ -I/usr/include/c++/v1

ALL += $(STRING_SORT_BENCH_ALL)

string_piece.o: string_piece.cc string_piece.h

$(STRING_SORT_BENCH_ALL): $(STRING_SORT_BENCH_OBJS)
	$(STRING_SORT_BENCH_CXX) $(STRING_SORT_BENCH_CXXFLAGS) $^ -o $@ -lc++

$(STRING_SORT_BENCH_OBJS): %.o: %.cc
	$(STRING_SORT_BENCH_CXX) -c $(STRING_SORT_BENCH_CXXFLAGS) -MF $@.d $< -o $@
