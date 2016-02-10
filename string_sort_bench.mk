STRING_SORT_BENCH_ALL := string_sort_bench
STRING_SORT_BENCH_OBJS := string_piece.o string_sort_bench.o
STRING_SORT_BENCH_CXXFLAGS := -g -W -Wall -std=c++11 -MMD -MP

ALL += $(STRING_SORT_BENCH_ALL)

string_piece.o: string_piece.cc string_piece.h

$(STRING_SORT_BENCH_ALL): $(STRING_SORT_BENCH_OBJS)
	$(CXX) $(STRING_SORT_BENCH_CXXFLAGS) $^ -o $@

$(STRING_SORT_BENCH_OBJS): %.o: %.cc
	$(CXX) -c $(STRING_SORT_BENCH_CXXFLAGS) -MF $@.d $< -o $@
