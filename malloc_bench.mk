all := malloc_bench
srcs := malloc_bench.cc kr_malloc.cc mmap_malloc.cc leak_malloc.cc my_malloc.cc
objs := $(srcs:.cc=.o)

ALL += $(all)
CLEAN += $(all) $(objs)

all: $(all)

$(objs): %.o: %.cc
	$(CXX) -std=c++11 -MMD -MP -c -g -O $< -o $@

malloc_bench: $(objs)
	$(CXX) -g -O $^ -o $@

-include $(objs:.o=.d)
