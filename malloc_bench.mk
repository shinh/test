all := malloc_bench
srcs := malloc_bench.cc kr_malloc.cc mmap_malloc.cc leak_malloc.cc
objs := $(srcs:.cc=.o)

ALL += $(all)
CLEAN += $(all) $(objs)

all: $(all)

$(objs): %.o: %.cc malloc_bench.h
	$(CXX) -std=c++11 -c -g -O $< -o $@

malloc_bench: $(objs)
	$(CXX) -g -O $^ -o $@

