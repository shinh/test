// https://twitter.com/e869120/status/1613672259931209728

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int ComputeScore(const std::vector<double>& vals) {
    std::vector<int> outs;
    for (int i = 0; i < vals.size(); ++i) {
        for (int j = i + 1; j < vals.size(); ++j) {
            int o = static_cast<int>(vals[i] * vals[j]);
            if (o > 0 && o <= 100) {
                outs.push_back(o);
            }
        }
    }
    std::sort(outs.begin(), outs.end());
    return std::unique(outs.begin(), outs.end()) - outs.begin();
}

int main() {
    int N = 17;
    std::mt19937 rng(42);
    std::vector<double> vals;
    for (int i = 0; i < N; ++i) {
        vals.push_back(std::uniform_real_distribution<double>(1, 10)(rng));
    }

    int score = ComputeScore(vals);
    std::cerr << "Initial score: " << score << "\n";

    std::vector<double> best;
    int best_score = score;
    int T = 10000000;
    double start_temp = 1e-1;
    double end_temp = 1e-5;
    for (int t = 0; t < T; ++t) {
        int c = std::uniform_int_distribution<int>(0, N)(rng);
        double d = std::uniform_real_distribution<double>(-1.5, 1.5)(rng);
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
    std::cerr << "{";
    for (double v : vals) {
        std::cerr << v << ",";
    }
    std::cerr << "}\n";
}
