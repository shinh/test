// https://twitter.com/e869120/status/1613672259931209728

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

double ComputeScore(const std::vector<double>& vals) {
    int score = 0;
    int done[102] = {};
    for (int i = 0; i < vals.size(); ++i) {
        for (int j = i + 1; j < vals.size(); ++j) {
            double od = vals[i] * vals[j];
            int o = static_cast<int>(od);
            if (o > 0 && o <= 100 && !done[o]) {
                ++score;
            }
            if (o >= 0 && o < 102) {
                ++done[o];
            }
        }
    }

    double bonus = 0.0;
    for (int i = 1; i <= 100; ++i) {
        if (done[i]) continue;
        if (done[i - 1] > 1) {
            bonus += (done[i - 1] - 1) * 0.01;
        }
        if (done[i + 1] > 1) {
            bonus += (done[i + 1] - 1) * 0.01;
        }
    }

    double penalty = 0.0;
#if 0
    for (int i = 1; i <= 100; ++i) {
        if (done[i] > 2) penalty += 0.01;
    }
#endif

    return score - penalty + bonus;
}

double SuggestChange(double v, double score, std::mt19937& rng) {
    double var = score > 90 ? 0.03 : 1.5;
    double mn = std::max<double>(-var, -v + 0.5);
    double mx = std::min<double>(var, 13 - v);
    double d = std::uniform_real_distribution<double>(mn, mx)(rng);
    double n = v + d;
    return std::min(n, 11.0);
}

int main(int argc, char* argv[]) {
    int N = 16;
    int seed = argc == 1 ? 42 : std::stoi(argv[1]);
    std::mt19937 rng(seed);
    std::vector<double> vals;
    for (int i = 0; i < N; ++i) {
        vals.push_back(std::uniform_real_distribution<double>(1, 10)(rng));
    }

    double score = ComputeScore(vals);
    std::cerr << "Initial score: " << score << "\n";

    std::vector<double> best;
    double best_score = score;
    int T = 10000000;
    double start_temp = 0.5;
    double end_temp = 1e-4;
    for (int t = 0; t < T; ++t) {
        int c = std::uniform_int_distribution<int>(0, N - 1)(rng);
        double o = vals[c];
        vals[c] = SuggestChange(vals[c], score, rng);
        double d = vals[c] - o;

        double next_score = ComputeScore(vals);
        if (next_score > best_score) {
            best = vals;
            best_score = next_score;
            std::cerr << "Best score updated at " << t << " " << best_score << " by " << d << "\n";
            if (best_score == 100) break;
        }

        double temp = start_temp + (end_temp - start_temp) * t / T;
        if (next_score > score || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp((next_score - score) / temp)) {
            score = next_score;
        } else {
            vals[c] = o;
        }
    }

    std::cerr << "Best score: " << best_score << "\n";
    std::sort(best.begin(), best.end());
    std::cerr << "{";
    for (double v : best) {
        std::cerr << v << ",";
    }
    std::cerr << "}\n";
}

// N=17
// Best score: 100
// {0.894668,1.92567,3.14647,4.11963,5.32202,5.88469,7.06001,7.6126,8.23402,8.47129,8.67602,8.96091,9.25522,9.57086,10.32,10.5001,10.7302}

// N=16
// Best score: 98
// {0.938699,2.0084,2.87645,4.08139,5.07487,6.11319,6.68365,7.92926,8.33935,8.79495,8.9944,9.20398,9.44163,9.73649,10.2465,10.601}

// N=16 seed=1688
// Best score: 99
// {0.915957,2.11986,3.03985,4.09351,5.2462,6.54203,6.71142,7.92987,8.40744,8.92736,9.17984,9.39551,9.5926,10.0081,10.4185,11}
