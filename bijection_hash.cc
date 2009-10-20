#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unordered_set>
#include <vector>

using namespace std;

struct P {
    int x, y;

    P(int x0, int y0) : x(x0), y(y0) {}

    bool operator==(const P& rhs) const {
        return x == rhs.x && y == rhs.y;
    }

    struct Mult {
        int operator()(const P& p) const {
            return p.x * p.y;
        }
    };

    struct Linear {
        int operator()(const P& p) const {
            return p.x + p.y * 17198911;
        }
    };

    struct Biject {
        int operator()(const P& p) const {
            return (p.x + p.y) * (p.x + p.y + 1) / 2 + p.x;
        }
    };
};

vector<int> x, y;

template <class Hash>
double run(const char* type) {
    clock_t start = clock();
    unordered_set<P, Hash> s;
    for (int i = 0; i < x.size(); i++) {
        s.insert(P(x[i], y[i]));
    }

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "%s: %lf\n", type, elapsed);
    return elapsed;
}

int main() {
    FILE* fp = fopen("bijection_hash.dat", "wb");

    int range = 100;
    for (int t = 0; t < 20; t++) {
        x.clear();
        y.clear();
        for (int i = 0; i < 1000 * 1000 * 2; i++) {
            x.push_back(rand() % range);
            y.push_back(rand() % range);
        }

        fprintf(stderr, "%d\n", range);
        double mt = run<P::Mult>("mult");
        double lt = run<P::Linear>("linear");
        double bt = run<P::Biject>("biject");

        fprintf(fp, "%d %lf %lf %lf\n", range, mt, lt, bt);

        range *= 2;
    }

    fclose(fp);
}
