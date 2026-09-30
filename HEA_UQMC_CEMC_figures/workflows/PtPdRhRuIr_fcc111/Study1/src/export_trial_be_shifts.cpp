#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

// Log-activity companion of export_trial_be_shifts.cpp.  The activity-domain
// selection is performed by plot_kde_neighbor_be_temperature_log_activity.py;
// this helper reconstructs the deterministic per-trial BE shifts for the
// selected trial IDs without modifying the original exporter.

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static uint64_t make_seed(uint64_t base, int a, int b=0, int c=0, int d=0) {
    uint64_t x = base;
    x ^= splitmix64(static_cast<uint64_t>(a) + 0x100000001b3ULL);
    x ^= splitmix64(static_cast<uint64_t>(b) + 0x9e3779b97f4a7c15ULL);
    x ^= splitmix64(static_cast<uint64_t>(c) + 0xbf58476d1ce4e5b9ULL);
    x ^= splitmix64(static_cast<uint64_t>(d) + 0x94d049bb133111ebULL);
    return splitmix64(x);
}
int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: export_trial_be_shifts_log_activity N_TRIALS BASE_SEED MEAN SIGMA" << std::endl;
        return 2;
    }
    const int n = std::atoi(argv[1]);
    const uint64_t base = static_cast<uint64_t>(std::strtoull(argv[2], nullptr, 10));
    const double mean = std::atof(argv[3]);
    const double sigma = std::atof(argv[4]);
    std::cout << "trial,be_shift_Ir,be_shift_Pd,be_shift_Pt,be_shift_Rh,be_shift_Ru" << std::endl;
    std::cout << std::setprecision(17);
    for (int trial=0; trial<n; ++trial) {
        std::mt19937_64 rng(make_seed(base, trial, 0, 202607, 505));
        std::normal_distribution<double> normal(mean, sigma);
        std::cout << trial;
        for (int k=0; k<5; ++k) std::cout << ',' << normal(rng);
        std::cout << '\n';
    }
}
