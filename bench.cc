#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <map>
#include <vector>
#include <assert.h>

#ifdef _MSC_VER

#include <hash_map>

#else

#include <tr1/unordered_map>

using namespace std::tr1;

#define hash_map unordered_map

#endif

using namespace std;

int main(int argc, char* argv[]) {
    map<int, int> m;
    hash_map<int, int> um;
    vector<int> v;

    static const int TOT = 1000000;
    const int N = atoi(argv[1]);
    for (int i = 0; i < N; i++) {
        v.push_back(rand());
        //v.push_back(i);
        //v.push_back(0);
    }
    {
        clock_t cl = clock();
        for (int t = 0; t < TOT/N; t++) {
            for (int i = 0; i < v.size(); i++) {
                um.insert(make_pair(v[i], i));
            }
            for (int i = 0; i < v.size(); i++) {
                um.erase(v[i]);
            }
        }
        assert(m.empty());
        printf("%.3f\n", ((double)clock() - cl) / CLOCKS_PER_SEC);
    }
    {
        clock_t cl = clock();
        for (int t = 0; t < TOT/N; t++) {
            for (int i = 0; i < v.size(); i++) {
                m.insert(make_pair(v[i], i));
            }
            for (int i = 0; i < v.size(); i++) {
                m.erase(v[i]);
            }
        }
        assert(m.empty());
        printf("%.3f\n", ((double)clock() - cl) / CLOCKS_PER_SEC);
    }
}
