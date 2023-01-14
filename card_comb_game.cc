// https://twitter.com/e869120/status/1613672259931209728

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int ComputeScore(const std::vector<double>& vals) {
    int score = 0;
    bool done[101] = {};
    for (int i = 0; i < vals.size(); ++i) {
        for (int j = i + 1; j < vals.size(); ++j) {
            int o = static_cast<int>(vals[i] * vals[j]);
            if (o > 0 && o <= 100 && !done[o]) {
                ++score;
                done[o] = true;
            }
        }
    }
    return score;
}

double SuggestChange(double v, std::mt19937& rng) {
    double mn = std::max<double>(-1.5, -v + 0.5);
    double mx = std::min<double>(1.5, 13 - v);
    double d = std::uniform_real_distribution<double>(mn, mx)(rng);
    return d;
}

int main(int argc, char* argv[]) {
    int N = 16;
    int seed = argc == 1 ? 42 : std::stoi(argv[1]);
    std::mt19937 rng(seed);
    std::vector<double> vals;
    for (int i = 0; i < N; ++i) {
        vals.push_back(std::uniform_real_distribution<double>(1, 10)(rng));
    }

    int score = ComputeScore(vals);
    std::cerr << "Initial score: " << score << "\n";

    std::vector<double> best;
    int best_score = score;
    int T = 10000000;
    double start_temp = 0.5;
    double end_temp = 1e-4;
    for (int t = 0; t < T; ++t) {
        int c = std::uniform_int_distribution<int>(0, N - 1)(rng);
        double d = SuggestChange(vals[c], rng);
        vals[c] += d;

        int next_score = ComputeScore(vals);
        if (next_score > best_score) {
            best = vals;
            best_score = next_score;
            std::cerr << "Best score updated at " << t << " " << best_score << "\n";
            if (best_score == 100) break;
        }

        double temp = start_temp + (end_temp - start_temp) * t / T;
        if (next_score > score || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp((next_score - score) / temp)) {
            score = next_score;
        } else {
            vals[c] -= d;
        }
    }

    std::cerr << "Best score: " << best_score << "\n";
    std::sort(vals.begin(), vals.end());
    std::cerr << "{";
    for (double v : vals) {
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
