#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

#include <gmp.h>
#include <gmpxx.h>

using namespace std;

int main(int argc, char* argv[]) {
    vector<mpz_class> v;
    v.push_back(1);
    v.push_back(1);
    int e = atoi(argv[1]);
    for (int i = 0; i < e; i++) {
        v.push_back(v[i] + v[i+1]);
    }
    //cout << v[e] << endl;
}
