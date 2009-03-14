/*
PentiumD 2.8GHz + VC2008
ave       2      5      7     10     12     16     20     32     64    128    256    512   1024
strlenANSI  894.5  511.8  386.5  296.8  265.8  222.5  207.0  171.8  148.5  132.8  128.8  128.8  129.0
strlenSSE2 1328.3  632.8  488.5  359.5  308.5  238.3  199.3  140.8   82.0   43.0   31.3   19.5   15.5
memchrANSI 1230.5  750.0  609.3  468.8  406.3  324.3  277.5  191.3  121.0   86.0   74.3   70.5   62.5
memchrSSE2 1496.0  703.3  527.3  390.5  332.0  254.0  207.0  152.5   93.8   50.8   31.3   23.3   19.5
strlenBLOG 1101.5  621.0  461.0  340.0  308.8  257.8  234.3  195.3  160.3  148.5  140.8  136.8  136.8

CoreDuo 1.8GHz + gcc 4.1.3
ave       2      5      7     10     12     16     20     32     64    128    256    512   1024
strlenANSI 2562.5 1155.0  900.0  687.5  615.0  522.5  460.0  382.5  310.0  275.0  257.5  250.0  247.5
strlenSSE2  695.0  342.5  272.5  207.5  180.0  142.5  120.0   87.5   55.0   37.5   27.5   17.5   17.5
memchrANSI 1412.5  765.0  612.5  465.0  407.5  330.0  280.0  205.0  140.0  110.0   90.0   82.5   80.0
memchrSSE2  805.0  390.0  305.0  225.0  197.5  157.5  132.5   97.5   62.5   40.0   30.0   22.5   20.0
strlenBLOG  815.0  480.0  387.5  307.5  280.0  242.5  220.0  187.5  160.0  140.0  130.0  125.0  125.0
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <time.h>
#include <vector>

#ifdef _WIN32
	#include <intrin.h>
	#define ALIGN(x) __declspec(align(x))
	#define bsf(x) (_BitScanForward(reinterpret_cast<unsigned long*>(&x), x), x)
	#define bsr(x) (_BitScanReverse(reinterpret_cast<unsigned long*>(&x), x), x)
#else
	#include <xmmintrin.h>
	#define ALIGN(x) __attribute__((aligned(x)))
	#define bsf(x) __builtin_ctz(x)
#endif

static const int funcNum = 5;

void *memchrSSE2(const void *ptr, int c, size_t len)
{
	const char *p = reinterpret_cast<const char*>(ptr);
	if (len >= 16) {
		__m128i c16 = _mm_set1_epi8(c);
		/* 16 byte alignment */
		size_t ip = reinterpret_cast<size_t>(p);
		size_t n = ip & 15;
		if (n > 0) {
			ip &= ~15;
			__m128i x = *(const __m128i*)ip;
			__m128i a = _mm_cmpeq_epi8(x, c16);
			size_t mask = _mm_movemask_epi8(a);
			mask &= -(1 << n);
			if (mask) {
				return (void*)(ip + bsf(mask));
			}
			n = 16 - n;
			len -= n;
			p += n;
		}
		while (len >= 32) {
			__m128i x = *(const __m128i*)&p[0];
			__m128i y = *(const __m128i*)&p[16];
			__m128i a = _mm_cmpeq_epi8(x, c16);
			__m128i b = _mm_cmpeq_epi8(y, c16);
			size_t mask = (_mm_movemask_epi8(b) << 16) | _mm_movemask_epi8(a);
			if (mask) {
				return (void*)(p + bsf(mask));
			}
			len -= 32;
			p += 32;
		}
	}
	while (len > 0) {
		if (*p == c) return (void*)p;
		p++;
		len--;
	}
	return 0;
}

size_t strlenSSE2(const char *p)
{
	const char *const top = p;
	__m128i c16 = _mm_set1_epi8(0);
	/* 16 byte alignment */
	size_t ip = reinterpret_cast<size_t>(p);
	size_t n = ip & 15;
	if (n > 0) {
		ip &= ~15;
		__m128i x = *(const __m128i*)ip;
		__m128i a = _mm_cmpeq_epi8(x, c16);
		size_t mask = _mm_movemask_epi8(a);
		mask &= -(1 << n);
		if (mask) {
			return bsf(mask) - n;
		}
		p += 16 - n;
	}
	for (;;) {
		__m128i x = *(const __m128i*)&p[0];
		__m128i y = *(const __m128i*)&p[16];
		__m128i a = _mm_cmpeq_epi8(x, c16);
		__m128i b = _mm_cmpeq_epi8(y, c16);
		size_t mask = (_mm_movemask_epi8(b) << 16) | _mm_movemask_epi8(a);
		if (mask) {
			return p + bsf(mask) - top;
		}
		p += 32;
	}
}

struct Result {
	int hit;
	int ret;
	double time;
	Result() {}
	Result(int hit, int ret, double time) : hit(hit), ret(ret), time(time) {}
	void put() const
	{
		printf("ret=%d(%.1f) time= %f usec\n", ret, ret / double(hit), time);
	}
};

void createTable(char *p, size_t num, int ave)
{
	int v = 0;
	int count = 0;
	for (size_t i = 0; i < num; i++) {
		v = 1;
		p[i] = v;
		if ((rand() % ave) == 0) p[i] = 0;
	}
	p[num - 1] = 0;
}

template<typename Func>
Result test(const char *top, size_t n, size_t count)
{
	int begin = clock();
	size_t ret = 0;
	int hit = 0;
	for (size_t i = 0; i < count; i++) {
		const char *p = top;
		int remain = n;
		while (remain > 0) {
			const char *q = Func::find(p, remain);
			if (q == 0) break;
			ret += q - p;
			hit++;
			remain -= q - p + 1;
			p = q + 1;
		}
	}
	return Result(hit, ret, (clock() - begin) * 1e6 / count / double(CLOCKS_PER_SEC));
}

struct FstrlenANSI {
	static inline const char *find(const char *p, size_t n)
	{
		return strlen(p) + p;
	}
};

struct FmemchrANSI {
	static inline const char *find(const char *p, size_t n)
	{
		return reinterpret_cast<const char*>(memchr(p, 0, n));
	}
};

struct FmemchrSSE2 {
	static inline const char *find(const char *p, size_t n)
	{
		return reinterpret_cast<const char*>(memchrSSE2(p, 0, n));
	}
};

struct FstrlenSSE2 {
	static inline const char *find(const char *p, size_t n)
	{
		return strlenSSE2(p) + p;
	}
};

int my_strlen(const char *s)
{ 
	int i = 0; 
	while (*s++) i++; 
	return i; 
} 
struct FstrlenBLOG {
	static inline const char *find(const char *p, size_t n)
	{
		return my_strlen(p) + p;
	}
};


#define NUM_OF_ARRAY(x) (sizeof(x)/sizeof(x[0]))

int main(int argc, char *argv[])
{
	int q = argc < 2 ? 70 : atoi(argv[1]);
	const size_t count = 4000;
	const size_t N = 100000;
	std::vector<char> v(N);

	typedef std::vector<Result> ResultVect;

	ResultVect rv[funcNum];

	char *begin = &v[0];

	const int aveTbl[] = { 2, 5, 7, 10, 12, 16, 20, 32, 64, 128, 256, 512, 1024 };

	for (size_t i = 0; i < NUM_OF_ARRAY(aveTbl); i++) {
		int ave = aveTbl[i];
		createTable(begin, N, ave);

		printf("test %d, %d\n", i, ave);
		Result ret;
		int hit;

		puts("strlenANSI");
		ret = test<FstrlenANSI>(begin, N, count);
		ret.put();
		rv[0].push_back(ret);
		hit = ret.hit;

		puts("strlenSSE2");
		ret = test<FstrlenSSE2>(begin, N, count);
		if (ret.hit != hit) { printf("ERROR!!! ok=%d, ng=%d\n", hit, ret.hit); }
		ret.put();
		rv[1].push_back(ret);

		puts("memchrANSI");
		ret = test<FmemchrANSI>(begin, N, count);
		if (ret.hit != hit) { printf("ERROR!!! ok=%d, ng=%d\n", hit, ret.hit); }
		ret.put();
		rv[2].push_back(ret);

		puts("memchrSSE2");
		ret = test<FmemchrSSE2>(begin, N, count);
		if (ret.hit != hit) { printf("ERROR!!! ok=%d, ng=%d\n", hit, ret.hit); }
		ret.put();
		rv[3].push_back(ret);

		puts("strlenBLOG");
		ret = test<FstrlenBLOG>(begin, N, count);
		if (ret.hit != hit) { printf("ERROR!!! ok=%d, ng=%d\n", hit, ret.hit); }
		ret.put();
		rv[4].push_back(ret);
	}

	puts("end");
	
	printf("ave  ");
	for (size_t i = 0; i < NUM_OF_ARRAY(aveTbl); i++) {
		printf("%6d ", aveTbl[i]);
	}
	printf("\n");
	static const char nameTbl[][16] = { "strlenANSI", "strlenSSE2", "memchrANSI", "memchrSSE2", "strlenBLOG" };
	for (int i = 0; i < funcNum; i++) {
		printf("%s ", nameTbl[i]);
		for (size_t j = 0; j < NUM_OF_ARRAY(aveTbl); j++) {
			printf("%6.1f ", rv[i][j].time);
		}
		printf("\n");
	}
	return 0;
}
