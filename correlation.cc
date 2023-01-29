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

int main() {
    {
        std::vector<double> xs, ys;
        for (int i = -100; i < 100; ++i) {
            xs.push_back(i);
            ys.push_back(i * 2 + 1);
        }
        std::cerr << correlation(xs, ys) << "\n";
    }

    {
        std::vector<double> xs, ys;
        for (int i = -100; i < 100; ++i) {
            xs.push_back(i);
            ys.push_back(-i * 2 + 1);
        }
        std::cerr << correlation(xs, ys) << "\n";
    }

    {
        std::mt19937 rng;
        std::vector<double> xs, ys;
        for (int i = 0; i < 1000000; ++i) {
            xs.push_back(std::uniform_real_distribution<double>(-100, 100)(rng));
            ys.push_back(std::uniform_real_distribution<double>(-100, 100)(rng));
        }
        std::cerr << correlation(xs, ys) << "\n";
    }

    {
        std::mt19937 rng;
        std::vector<double> xs, ys;
        for (int i = 0; i < 1000000; ++i) {
            xs.push_back(std::normal_distribution<double>(-100, 100)(rng));
            ys.push_back(std::normal_distribution<double>(-100, 100)(rng));
        }
        std::cerr << correlation(xs, ys) << "\n";
    }
}
