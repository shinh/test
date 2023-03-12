#include <cmath>
#include <iostream>
#include <random>
#include <vector>

double average(const std::vector<double>& xs) {
    double sum = 0.0;
    for (double x : xs) sum += x;
    return sum / xs.size();
}

double stddev(const std::vector<double>& xs) {
    double m = average(xs);
    double sum = 0.0;
    for (double x : xs) {
        double d = x - m;
        sum += d * d;
    }
    return sqrt(sum / xs.size());
}

double correlation(const std::vector<double>& xs, const std::vector<double>& ys) {
    double xm = average(xs);
    double ym = average(ys);
    double cov = 0.0;
    for (int i = 0; i < xs.size(); ++i) {
        cov += (xs[i] - xm) * (ys[i] - ym);
    }
    cov /= xs.size();
    return cov / (stddev(xs) * stddev(ys));
}

double iq(const std::vector<int>& vs) {
    double q = 0;
    for (int v : vs) {
        q += v;
    }
    return q;
}

int main() {
    {
        std::mt19937 rng;

        std::vector<double> xs, ys;
        for (int t = 0; t < 10000; ++t) {
            std::vector<int> parents[2];
            for (int i = 0; i < 1; ++i) {
                parents[0].push_back(std::uniform_int_distribution<int>(0, 1)(rng));
                parents[1].push_back(std::uniform_int_distribution<int>(0, 1)(rng));
            }

            std::vector<int> twins[2];
            for (int i = 0; i < parents[0].size(); ++i) {
                twins[0].push_back(parents[std::uniform_int_distribution<int>(0, 1)(rng)][i]);
                twins[1].push_back(parents[std::uniform_int_distribution<int>(0, 1)(rng)][i]);
            }

            double p0 = iq(parents[0]);
            double p1 = iq(parents[1]);
            double x = iq(twins[0]);
            double y = iq(twins[1]);

            std::cerr << "t=" << t << " x=" << x << " y=" << y << " p0=" << p0 << " p1=" << p1 << "\n";

            xs.push_back(x);
            ys.push_back(y);
        }

        std::cerr << correlation(xs, ys) << "\n";
    }
}
