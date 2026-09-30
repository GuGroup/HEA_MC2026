// uq_cemc_mpi.cpp
// C++17/MPI Monte Carlo uncertainty + CEMC annealing for Fe-Co-Ni-Pd-Pt fcc(100) slabs.
// v1.0: writes selected-temperature per-composition predicted activities for map/scatter diagnostics; keeps v1.0 seed-only production mode.
// Build with MPI when available: mpicxx -std=c++17 -O3 -DUSE_MPI -o uq_cemc_mpi uq_cemc_mpi.cpp
// Serial fallback: g++ -std=c++17 -O3 -o uq_cemc_mpi uq_cemc_mpi.cpp

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace fs = std::filesystem;
static constexpr double KB_EV_PER_K = 8.617333262145e-5;

static std::string trim(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) e--;
    return s.substr(b, e-b);
}

static bool starts_with(const std::string &s, const std::string &p) {
    return s.rfind(p, 0) == 0;
}

static std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') { quoted = !quoted; continue; }
        if (c == ',' && !quoted) { out.push_back(trim(cur)); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(trim(cur));
    return out;
}

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

static uint64_t uniform_uint_inclusive(std::mt19937_64 &rng, uint64_t hi) {
    // Deterministic bounded integer draw, independent of std::uniform_int_distribution.
    // Returns an integer in [0, hi].
    if (hi == std::numeric_limits<uint64_t>::max()) return rng();
    const uint64_t bound = hi + 1ULL;
    const uint64_t threshold = (uint64_t)(-bound) % bound;
    for (;;) {
        uint64_t r = rng();
        if (r >= threshold) return r % bound;
    }
}

template <class T>
static void deterministic_shuffle(std::vector<T> &v, std::mt19937_64 &rng) {
    // Fisher-Yates shuffle using the deterministic bounded draw above. This makes
    // saved random_slab_seed + element counts sufficient to reproduce a random slab.
    if (v.size() <= 1) return;
    for (size_t i = v.size() - 1; i > 0; --i) {
        size_t j = static_cast<size_t>(uniform_uint_inclusive(rng, static_cast<uint64_t>(i)));
        std::swap(v[i], v[j]);
    }
}

struct Config {
    std::string ce_export;
    std::string schedule_export;
    std::string activity_model;
    std::string composition_csv;
    std::string experimental_activity_csv;
    std::string output_dir = "uq_output";
    int ml_index = 1;
    int n_trials = 2;
    int n_runs = 2;
    int max_compositions = -1;
    uint64_t random_seed = 20260706ULL;
    double comp_error_mean = 0.000347;
    double comp_error_sigma = 0.046858;
    std::string comp_error_units = "fraction"; // fraction or percent
    double be_error_mean = -0.0575;
    double be_error_sigma = 0.4875;
    bool write_structures = false;
    bool write_random_structures = false;
    bool write_random_seeds = true;
    bool write_predicted_activities = true;
    bool predicted_better_is_higher = true;
    bool experimental_better_is_lower = true;
    int summary_every_trials = 0; // 0=end only; N=also every N trials
};

static Config read_config(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open config: " + path);
    Config cfg;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty() || line.front() == '[') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq+1));
        auto unquote = [](std::string v) {
            v = trim(v);
            if (v.size() >= 2 && ((v.front()=='"' && v.back()=='"') || (v.front()=='\'' && v.back()=='\'')))
                v = v.substr(1, v.size()-2);
            return v;
        };
        val = unquote(val);
        auto as_bool = [](const std::string &v) {
            std::string x = v;
            std::transform(x.begin(), x.end(), x.begin(), ::tolower);
            return x == "1" || x == "true" || x == "yes" || x == "y";
        };
        try {
            if (key == "ce_export") cfg.ce_export = val;
            else if (key == "schedule_export") cfg.schedule_export = val;
            else if (key == "activity_model") cfg.activity_model = val;
            else if (key == "composition_csv") cfg.composition_csv = val;
            else if (key == "experimental_activity_csv") cfg.experimental_activity_csv = val;
            else if (key == "output_dir") cfg.output_dir = val;
            else if (key == "ml_index") cfg.ml_index = std::stoi(val);
            else if (key == "n_trials") cfg.n_trials = std::stoi(val);
            else if (key == "n_runs") cfg.n_runs = std::stoi(val);
            else if (key == "max_compositions") cfg.max_compositions = std::stoi(val);
            else if (key == "random_seed") cfg.random_seed = static_cast<uint64_t>(std::stoull(val));
            else if (key == "comp_error_mean") cfg.comp_error_mean = std::stod(val);
            else if (key == "comp_error_sigma") cfg.comp_error_sigma = std::stod(val);
            else if (key == "comp_error_units") cfg.comp_error_units = val;
            else if (key == "be_error_mean") cfg.be_error_mean = std::stod(val);
            else if (key == "be_error_sigma") cfg.be_error_sigma = std::stod(val);
            else if (key == "write_structures") cfg.write_structures = as_bool(val);
            else if (key == "write_random_structures") cfg.write_random_structures = as_bool(val);
            else if (key == "write_random_seeds") cfg.write_random_seeds = as_bool(val);
            else if (key == "write_predicted_activities") cfg.write_predicted_activities = as_bool(val);
            else if (key == "predicted_better_is_higher") cfg.predicted_better_is_higher = as_bool(val);
            else if (key == "experimental_better_is_lower") cfg.experimental_better_is_lower = as_bool(val);
            else if (key == "summary_every_trials") cfg.summary_every_trials = std::stoi(val);
        } catch (const std::exception &e) {
            throw std::runtime_error("Bad config line " + std::to_string(lineno) + ": " + line + " (" + e.what() + ")");
        }
    }
    if (cfg.ce_export.empty() || cfg.schedule_export.empty() || cfg.activity_model.empty() ||
        cfg.composition_csv.empty() || cfg.experimental_activity_csv.empty()) {
        throw std::runtime_error("Config must define ce_export, schedule_export, activity_model, composition_csv, experimental_activity_csv");
    }
    return cfg;
}

template <typename T>
static std::vector<T> read_vec(std::istream &in, size_t n) {
    std::vector<T> v(n);
    for (size_t i=0; i<n; ++i) {
        if (!(in >> v[i])) throw std::runtime_error("Unexpected EOF while reading vector");
    }
    return v;
}

struct CEData {
    int q = 0;
    int vacancy_code = -1;
    int n_sites = 0;
    int n_metal_sites = 0;
    int pair_shell_count = 0;
    int n_triplet_geometries = 0;
    std::vector<std::string> species;
    std::vector<int> atomic_numbers;
    std::vector<int> metal_sites;
    std::vector<int> active_sites;
    std::vector<int> surface_sites;
    double zero = 0.0;
    std::vector<double> point;          // q
    std::vector<double> pair_coeff;     // pair_shell_count*q*q, shell included, shell 0 unused
    std::vector<int> pair_i, pair_j, pair_shell;
    std::vector<double> triplet_coeff;  // G*q*q*q
    std::vector<int> triplet_i, triplet_j, triplet_k, triplet_gid;
    std::vector<int> site_pair_ptr, site_pair_indices;
    std::vector<int> site_triplet_ptr, site_triplet_indices;
    std::unordered_map<std::string, int> species_to_code;

    double pairC(int shell, int a, int b) const {
        size_t idx = (static_cast<size_t>(shell) * q + a) * q + b;
        return pair_coeff[idx];
    }
    double tripletC(int gid, int a, int b, int c) const {
        size_t idx = (((static_cast<size_t>(gid) * q + a) * q + b) * q + c);
        return triplet_coeff[idx];
    }
};

static CEData read_ce_export(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open CE export: " + path);
    std::string magic;
    in >> magic;
    if (magic != "FCC_CE_EXPORT_V1") throw std::runtime_error("Bad CE export magic: " + magic);
    CEData ce;
    std::string key;
    while (in >> key) {
        if (key == "n_species") { in >> ce.q; }
        else if (key == "species") {
            if (ce.q <= 0) throw std::runtime_error("species before n_species");
            ce.species = read_vec<std::string>(in, ce.q);
            ce.species_to_code.clear();
            for (int i=0;i<ce.q;i++) ce.species_to_code[ce.species[i]] = i;
        }
        else if (key == "atomic_numbers") ce.atomic_numbers = read_vec<int>(in, ce.q);
        else if (key == "vacancy_code") in >> ce.vacancy_code;
        else if (key == "n_sites") in >> ce.n_sites;
        else if (key == "n_metal_sites") in >> ce.n_metal_sites;
        else if (key == "metal_sites") ce.metal_sites = read_vec<int>(in, ce.n_metal_sites);
        else if (key == "n_active_sites") {
            int n; in >> n;
            std::string k2; in >> k2; if (k2 != "active_sites") throw std::runtime_error("Expected active_sites");
            ce.active_sites = read_vec<int>(in, n);
        }
        else if (key == "n_surface_sites") {
            int n; in >> n;
            std::string k2; in >> k2; if (k2 != "surface_sites") throw std::runtime_error("Expected surface_sites");
            ce.surface_sites = read_vec<int>(in, n);
        }
        else if (key == "zero") in >> ce.zero;
        else if (key == "point") ce.point = read_vec<double>(in, ce.q);
        else if (key == "pair_shape") {
            int s,q1,q2; in >> s >> q1 >> q2;
            if (q1 != ce.q || q2 != ce.q) throw std::runtime_error("pair_shape species mismatch");
            ce.pair_shell_count = s;
            ce.pair_coeff = read_vec<double>(in, static_cast<size_t>(s)*ce.q*ce.q);
        }
        else if (key == "n_pairs") {
            int n; in >> n;
            std::string k2; in >> k2; if (k2 != "pair_i") throw std::runtime_error("Expected pair_i");
            ce.pair_i = read_vec<int>(in, n);
            in >> k2; if (k2 != "pair_j") throw std::runtime_error("Expected pair_j");
            ce.pair_j = read_vec<int>(in, n);
            in >> k2; if (k2 != "pair_shell") throw std::runtime_error("Expected pair_shell");
            ce.pair_shell = read_vec<int>(in, n);
        }
        else if (key == "triplet_shape") {
            int g,q1,q2,q3; in >> g >> q1 >> q2 >> q3;
            if (q1 != ce.q || q2 != ce.q || q3 != ce.q) throw std::runtime_error("triplet_shape species mismatch");
            ce.n_triplet_geometries = g;
            ce.triplet_coeff = read_vec<double>(in, static_cast<size_t>(std::max(g,0))*ce.q*ce.q*ce.q);
        }
        else if (key == "n_triplets") {
            int n; in >> n;
            std::string k2; in >> k2; if (k2 != "triplet_i") throw std::runtime_error("Expected triplet_i");
            ce.triplet_i = read_vec<int>(in, n);
            in >> k2; if (k2 != "triplet_j") throw std::runtime_error("Expected triplet_j");
            ce.triplet_j = read_vec<int>(in, n);
            in >> k2; if (k2 != "triplet_k") throw std::runtime_error("Expected triplet_k");
            ce.triplet_k = read_vec<int>(in, n);
            in >> k2; if (k2 != "triplet_gid") throw std::runtime_error("Expected triplet_gid");
            ce.triplet_gid = read_vec<int>(in, n);
        }
        else if (key == "site_pair_ptr") ce.site_pair_ptr = read_vec<int>(in, ce.n_sites + 1);
        else if (key == "site_pair_indices") { int n; in >> n; ce.site_pair_indices = read_vec<int>(in, n); }
        else if (key == "site_triplet_ptr") ce.site_triplet_ptr = read_vec<int>(in, ce.n_sites + 1);
        else if (key == "site_triplet_indices") { int n; in >> n; ce.site_triplet_indices = read_vec<int>(in, n); }
        else {
            throw std::runtime_error("Unknown CE export key: " + key);
        }
    }
    if (ce.q <= 0 || ce.n_sites <= 0 || ce.n_metal_sites <= 0 || ce.vacancy_code < 0) throw std::runtime_error("Incomplete CE export");
    if (static_cast<int>(ce.point.size()) != ce.q) throw std::runtime_error("CE point coeff missing");
    if (static_cast<int>(ce.atomic_numbers.size()) != ce.q) throw std::runtime_error("CE atomic numbers missing");
    return ce;
}

struct ScheduleSegment {
    int profile = 0; // 0 hold, 1 linear, 2 exponential
    double start = 300.0;
    double stop = 300.0;
    int64_t steps = 0;
};

struct SchedulePlan {
    std::vector<double> target_temperatures;
    std::vector<ScheduleSegment> segments;
};

static SchedulePlan read_schedule_export(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open schedule export: " + path);
    std::string magic;
    in >> magic;
    if (magic != "SCHEDULE_SNAPSHOTS_V1") throw std::runtime_error("Bad schedule export magic: " + magic);
    SchedulePlan plan;
    std::string key;
    int n_targets = 0;
    in >> key >> n_targets;
    if (key != "n_targets" || n_targets <= 0) throw std::runtime_error("Bad schedule n_targets");
    in >> key;
    if (key != "target_temperatures") throw std::runtime_error("Expected target_temperatures in schedule export");
    plan.target_temperatures = read_vec<double>(in, n_targets);
    int nseg = 0;
    in >> key >> nseg;
    if (key != "n_segments" || nseg <= 0) throw std::runtime_error("Expected n_segments in schedule export");
    for (int s=0; s<nseg; ++s) {
        std::string segkey;
        ScheduleSegment seg;
        in >> segkey >> seg.profile >> seg.start >> seg.stop >> seg.steps;
        if (segkey != "segment") throw std::runtime_error("Expected segment in schedule export");
        if (seg.steps <= 0) throw std::runtime_error("Schedule segment has nonpositive steps");
        plan.segments.push_back(seg);
    }
    return plan;
}


struct ActivityModel {
    std::string site_type = "top"; // top or fcc
    std::vector<std::string> elements;
    std::unordered_map<std::string,int> element_to_index;
    std::vector<double> zone1_element, zone2, zone3, zone4, zone5;
    std::unordered_map<std::string,double> zone1_combo_coeff;
    std::unordered_map<std::string,double> zone3_combo_coeff;
    double intercept = 0.0;
    double e_opt = -0.11;
    double activity_temperature = 298.0;
    std::vector<int> zone1_ptr, zone1_indices;
    std::vector<int> zone2_ptr, zone2_indices;
    std::vector<int> zone3_ptr, zone3_indices;
    std::vector<int> zone4_ptr, zone4_indices;
    std::vector<int> zone5_ptr, zone5_indices;
};

static std::vector<double> zeros5() { return std::vector<double>(5, 0.0); }

static std::vector<int> read_ptr_vec(std::istream &in, int n_sites) {
    return read_vec<int>(in, static_cast<size_t>(n_sites)+1);
}

static ActivityModel read_activity_model(const std::string &path, const CEData &ce) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open activity model: " + path);
    std::string magic;
    in >> magic;
    if (magic != "ACTIVITY_MODEL_H_V1") throw std::runtime_error("Bad activity model magic: " + magic);
    ActivityModel am;
    am.zone1_element = zeros5();
    am.zone2 = zeros5();
    am.zone3 = zeros5();
    am.zone4 = zeros5();
    am.zone5 = zeros5();
    std::string key;
    int n_sites = -1;
    while (in >> key) {
        if (key == "site_type") in >> am.site_type;
        else if (key == "elements") {
            am.elements = read_vec<std::string>(in, 5);
            am.element_to_index.clear();
            for (int i=0;i<5;i++) am.element_to_index[am.elements[i]] = i;
        }
        else if (key == "intercept") in >> am.intercept;
        else if (key == "e_opt") in >> am.e_opt;
        else if (key == "activity_temperature") in >> am.activity_temperature;
        else if (key == "zone1_element") am.zone1_element = read_vec<double>(in, 5);
        else if (key == "zone2") am.zone2 = read_vec<double>(in, 5);
        else if (key == "zone3") am.zone3 = read_vec<double>(in, 5);
        else if (key == "zone4") am.zone4 = read_vec<double>(in, 5);
        else if (key == "zone5") am.zone5 = read_vec<double>(in, 5);
        else if (key == "zone1_combos") {
            int n; in >> n;
            for (int i=0;i<n;i++) {
                std::string combo; double coeff;
                in >> combo >> coeff;
                am.zone1_combo_coeff[combo] = coeff;
            }
        }
        else if (key == "zone3_combos") {
            int n; in >> n;
            for (int i=0;i<n;i++) {
                std::string combo; double coeff;
                in >> combo >> coeff;
                am.zone3_combo_coeff[combo] = coeff;
            }
        }
        else if (key == "n_activity_sites") in >> n_sites;
        else if (key == "zone1_ptr") { if (n_sites < 0) throw std::runtime_error("n_activity_sites must precede zone pointers"); am.zone1_ptr = read_ptr_vec(in, n_sites); }
        else if (key == "zone1_indices") { int n; in >> n; am.zone1_indices = read_vec<int>(in, n); }
        else if (key == "zone2_ptr") { if (n_sites < 0) throw std::runtime_error("n_activity_sites must precede zone pointers"); am.zone2_ptr = read_ptr_vec(in, n_sites); }
        else if (key == "zone2_indices") { int n; in >> n; am.zone2_indices = read_vec<int>(in, n); }
        else if (key == "zone3_ptr") { if (n_sites < 0) throw std::runtime_error("n_activity_sites must precede zone pointers"); am.zone3_ptr = read_ptr_vec(in, n_sites); }
        else if (key == "zone3_indices") { int n; in >> n; am.zone3_indices = read_vec<int>(in, n); }
        else if (key == "zone4_ptr") { if (n_sites < 0) throw std::runtime_error("n_activity_sites must precede zone pointers"); am.zone4_ptr = read_ptr_vec(in, n_sites); }
        else if (key == "zone4_indices") { int n; in >> n; am.zone4_indices = read_vec<int>(in, n); }
        else if (key == "zone5_ptr") { if (n_sites < 0) throw std::runtime_error("n_activity_sites must precede zone pointers"); am.zone5_ptr = read_ptr_vec(in, n_sites); }
        else if (key == "zone5_indices") { int n; in >> n; am.zone5_indices = read_vec<int>(in, n); }
        else throw std::runtime_error("Unknown activity model key: " + key);
    }
    if (am.elements.size() != 5) throw std::runtime_error("Activity model requires exactly five elements");
    if (am.site_type != "top" && am.site_type != "fcc" && am.site_type != "hollow" && am.site_type != "bridge" && am.site_type != "fcc100_hollow" && am.site_type != "fcc100_bridge") throw std::runtime_error("Unsupported activity site_type: " + am.site_type);
    int n = static_cast<int>(am.zone1_ptr.size()) - 1;
    if (n <= 0 || am.zone1_indices.empty()) throw std::runtime_error("Activity model missing zone1 topology");
    auto check_ptr = [&](const std::vector<int> &ptr, const std::string &name) {
        if (!ptr.empty() && (int)ptr.size() != n+1) throw std::runtime_error("Activity model " + name + " pointer size mismatch");
    };
    check_ptr(am.zone2_ptr, "zone2");
    check_ptr(am.zone3_ptr, "zone3");
    check_ptr(am.zone4_ptr, "zone4");
    check_ptr(am.zone5_ptr, "zone5");
    return am;
}

struct CompositionRow {
    int row_idx = 0;
    int col_idx = 0;
    std::array<double,5> fraction{}; // Fe,Co,Ni,Pd,Pt order in code below
    double experimental = 0.0;
};

static std::vector<CompositionRow> read_compositions_and_activity(const std::string &comp_path, const std::string &act_path, int max_compositions) {
    std::ifstream cinp(comp_path), ainp(act_path);
    if (!cinp) throw std::runtime_error("Could not open composition CSV: " + comp_path);
    if (!ainp) throw std::runtime_error("Could not open activity CSV: " + act_path);
    std::string line;
    if (!std::getline(cinp, line)) throw std::runtime_error("Empty composition CSV");
    auto h = split_csv_line(line);
    std::unordered_map<std::string,int> hc;
    for (int i=0;i<(int)h.size();i++) hc[h[i]] = i;
    const std::array<std::string,5> ratio_cols = {"Fe_ratio","Co_ratio","Ni_ratio","Pd_ratio","Pt_ratio"};
    for (auto &c : ratio_cols) if (!hc.count(c)) throw std::runtime_error("Missing composition column: " + c);
    if (!hc.count("Row_Idx") || !hc.count("Col_Idx")) throw std::runtime_error("Missing Row_Idx/Col_Idx in composition CSV");
    std::map<std::pair<int,int>, std::array<double,5>> comp_map;
    while (std::getline(cinp, line)) {
        if (trim(line).empty()) continue;
        auto v = split_csv_line(line);
        int r = std::stoi(v[hc["Row_Idx"]]);
        int c = std::stoi(v[hc["Col_Idx"]]);
        std::array<double,5> f{};
        double sum = 0.0;
        for (int k=0;k<5;k++) { f[k] = std::stod(v[hc[ratio_cols[k]]]); sum += f[k]; }
        if (sum > 1.5) for (double &x : f) x /= 100.0;
        double s = std::accumulate(f.begin(), f.end(), 0.0);
        if (s <= 0) throw std::runtime_error("Bad composition row with zero sum");
        for (double &x : f) x /= s;
        comp_map[{r,c}] = f;
    }
    if (!std::getline(ainp, line)) throw std::runtime_error("Empty activity CSV");
    h = split_csv_line(line);
    std::unordered_map<std::string,int> ha;
    for (int i=0;i<(int)h.size();i++) ha[h[i]] = i;
    if (!ha.count("Row_Idx") || !ha.count("Col_Idx") || !ha.count("Activity")) throw std::runtime_error("Missing Row_Idx/Col_Idx/Activity in activity CSV");
    std::vector<CompositionRow> rows;
    while (std::getline(ainp, line)) {
        if (trim(line).empty()) continue;
        auto v = split_csv_line(line);
        int r = std::stoi(v[ha["Row_Idx"]]);
        int c = std::stoi(v[ha["Col_Idx"]]);
        auto it = comp_map.find({r,c});
        if (it == comp_map.end()) throw std::runtime_error("Activity row has no composition row");
        CompositionRow row;
        row.row_idx = r; row.col_idx = c; row.fraction = it->second; row.experimental = std::stod(v[ha["Activity"]]);
        rows.push_back(row);
        if (max_compositions > 0 && (int)rows.size() >= max_compositions) break;
    }
    return rows;
}

static double total_energy(const CEData &ce, const std::vector<int> &occ) {
    double e = ce.zero * static_cast<double>(ce.n_metal_sites);
    for (int code : occ) e += ce.point[code];
    for (size_t p=0;p<ce.pair_i.size();p++) e += ce.pairC(ce.pair_shell[p], occ[ce.pair_i[p]], occ[ce.pair_j[p]]);
    for (size_t t=0;t<ce.triplet_i.size();t++) e += ce.tripletC(ce.triplet_gid[t], occ[ce.triplet_i[t]], occ[ce.triplet_j[t]], occ[ce.triplet_k[t]]);
    return e;
}

static inline int swapped_code(int idx, int a, int b, int code_a, int code_b, int cur) {
    if (idx == a) return code_b;
    if (idx == b) return code_a;
    return cur;
}

static double swap_delta(
    const CEData &ce, int a, int b, const std::vector<int> &occ,
    std::vector<int> &pair_marks, std::vector<int> &triplet_marks, int token
) {
    int ca = occ[a], cb = occ[b];
    if (ca == cb) return 0.0;
    double d = 0.0; // canonical point terms cancel
    int actives[2] = {a,b};
    for (int active : actives) {
        for (int ptr = ce.site_pair_ptr[active]; ptr < ce.site_pair_ptr[active+1]; ++ptr) {
            int p = ce.site_pair_indices[ptr];
            if (pair_marks[p] == token) continue;
            pair_marks[p] = token;
            int i = ce.pair_i[p], j = ce.pair_j[p];
            int oi = occ[i], oj = occ[j];
            int ni = swapped_code(i, a, b, ca, cb, oi);
            int nj = swapped_code(j, a, b, ca, cb, oj);
            int sh = ce.pair_shell[p];
            d += ce.pairC(sh, ni, nj) - ce.pairC(sh, oi, oj);
        }
    }
    for (int active : actives) {
        for (int ptr = ce.site_triplet_ptr[active]; ptr < ce.site_triplet_ptr[active+1]; ++ptr) {
            int t = ce.site_triplet_indices[ptr];
            if (triplet_marks[t] == token) continue;
            triplet_marks[t] = token;
            int i = ce.triplet_i[t], j = ce.triplet_j[t], k = ce.triplet_k[t];
            int oi = occ[i], oj = occ[j], ok = occ[k];
            int ni = swapped_code(i, a, b, ca, cb, oi);
            int nj = swapped_code(j, a, b, ca, cb, oj);
            int nk = swapped_code(k, a, b, ca, cb, ok);
            int gid = ce.triplet_gid[t];
            d += ce.tripletC(gid, ni, nj, nk) - ce.tripletC(gid, oi, oj, ok);
        }
    }
    return d;
}

static double schedule_temperature(const ScheduleSegment &seg, int64_t local_step) {
    if (seg.profile == 0 || seg.steps <= 1) return seg.start;
    double f = static_cast<double>(local_step) / static_cast<double>(seg.steps - 1);
    if (seg.profile == 1) return seg.start + f * (seg.stop - seg.start);
    if (seg.start <= 0 || seg.stop <= 0) return seg.stop;
    return seg.start * std::exp(f * std::log(seg.stop / seg.start));
}

struct Snapshot {
    std::vector<int> occ;
    double energy = 0.0;
    int64_t global_step = 0;
    double actual_temperature = 0.0;
    int64_t attempted = 0;
    int64_t accepted = 0;
    double temperature_error = std::numeric_limits<double>::infinity();
    bool captured = false;
};

struct MCOutcome {
    std::vector<int> occ;
    std::vector<Snapshot> snapshots;
    double energy = 0.0;
    int64_t attempted = 0;
    int64_t accepted = 0;
};

static void capture_snapshot(MCOutcome &out, size_t ti, double target_T, double T, int64_t global_step) {
    double diff = std::abs(T - target_T);
    if (!out.snapshots[ti].captured || diff + 1e-12 < out.snapshots[ti].temperature_error) {
        out.snapshots[ti].occ = out.occ;
        out.snapshots[ti].energy = out.energy;
        out.snapshots[ti].global_step = global_step;
        out.snapshots[ti].actual_temperature = T;
        out.snapshots[ti].attempted = out.attempted;
        out.snapshots[ti].accepted = out.accepted;
        out.snapshots[ti].temperature_error = diff;
        out.snapshots[ti].captured = true;
    }
}

static void maybe_capture_crossing(
    MCOutcome &out, const SchedulePlan &plan, double previous_T, double T, bool have_previous, int64_t global_step
) {
    for (size_t ti=0; ti<plan.target_temperatures.size(); ++ti) {
        double target = plan.target_temperatures[ti];
        bool exact = std::abs(T - target) <= 1e-9;
        bool crossed = false;
        if (have_previous) {
            crossed = (previous_T >= target && T <= target) || (previous_T <= target && T >= target);
        }
        if (exact || crossed) capture_snapshot(out, ti, target, T, global_step);
    }
}

static MCOutcome run_cemc_snapshots(const CEData &ce, const std::vector<int> &initial, const SchedulePlan &plan, std::mt19937_64 &rng) {
    MCOutcome out;
    out.occ = initial;
    out.energy = total_energy(ce, out.occ);
    out.snapshots.resize(plan.target_temperatures.size());
    std::uniform_int_distribution<int> active_dist(0, static_cast<int>(ce.active_sites.size()) - 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<int> pair_marks(ce.pair_i.size(), 0), triplet_marks(ce.triplet_i.size(), 0);
    int token = 1;
    int64_t global_step = 0;
    bool have_previous_T = false;
    double previous_T = 0.0;
    for (const auto &seg : plan.segments) {
        for (int64_t step=0; step<seg.steps; ++step) {
            double T = schedule_temperature(seg, step);
            int a=-1,b=-1;
            for (int tries=0; tries<64; ++tries) {
                a = ce.active_sites[active_dist(rng)];
                b = ce.active_sites[active_dist(rng)];
                if (a != b && out.occ[a] != out.occ[b]) break;
            }
            if (!(a == b || out.occ[a] == out.occ[b])) {
                double de = swap_delta(ce, a, b, out.occ, pair_marks, triplet_marks, token++);
                bool accept = de <= 0.0;
                if (!accept && T > 0.0) {
                    double p = std::exp(-de / (KB_EV_PER_K * T));
                    accept = uni(rng) < p;
                }
                out.attempted++;
                if (accept) {
                    std::swap(out.occ[a], out.occ[b]);
                    out.energy += de;
                    out.accepted++;
                }
                if (token > 2000000000) {
                    std::fill(pair_marks.begin(), pair_marks.end(), 0);
                    std::fill(triplet_marks.begin(), triplet_marks.end(), 0);
                    token = 1;
                }
            }
            global_step++;
            // Preserve the original schedule.yaml. Capture a snapshot only when the
            // cooling/heating trajectory crosses a requested target temperature, or
            // when a hold segment exactly matches the target. This avoids running
            // 18 independent anneals and avoids copying the full slab every step.
            maybe_capture_crossing(out, plan, previous_T, T, have_previous_T, global_step);
            previous_T = T;
            have_previous_T = true;
        }
    }
    // Fallback for unusual schedules that never cross a target: write the final state.
    for (size_t ti=0; ti<plan.target_temperatures.size(); ++ti) {
        if (!out.snapshots[ti].captured) {
            capture_snapshot(out, ti, plan.target_temperatures[ti], previous_T, global_step);
        }
    }
    return out;
}


static int code_to_activity_element(const ActivityModel &am, const CEData &ce, int code) {
    const std::string &sym = ce.species[code];
    auto it = am.element_to_index.find(sym);
    return it == am.element_to_index.end() ? -1 : it->second;
}

static std::string sorted_combo_key(const ActivityModel &am, const CEData &ce, const std::vector<int> &occ, const std::vector<int> &sites) {
    std::vector<int> idx;
    idx.reserve(sites.size());
    for (int site : sites) {
        int ai = code_to_activity_element(am, ce, occ[site]);
        if (ai < 0) return "";
        idx.push_back(ai);
    }
    std::sort(idx.begin(), idx.end());
    std::string key;
    for (int ai : idx) key += am.elements[ai];
    return key;
}

static double add_zone_element_terms(const CEData &ce, const ActivityModel &am, const std::vector<int> &occ,
                                     const std::vector<int> &ptr, const std::vector<int> &idx,
                                     const std::vector<double> &coeff, int sidx) {
    if (ptr.empty()) return 0.0;
    double e = 0.0;
    for (int p=ptr[sidx]; p<ptr[sidx+1]; ++p) {
        int ai = code_to_activity_element(am, ce, occ[idx[p]]);
        if (ai >= 0) e += coeff[ai];
    }
    return e;
}

static bool binding_energy_for_site(
    const CEData &ce, const ActivityModel &am, const std::vector<int> &occ,
    const std::array<double,5> &be_shift, int sidx, double &e,
    std::string *zone1_combo_out = nullptr
) {
        e = am.intercept;
        std::vector<int> z1sites;
        for (int p=am.zone1_ptr[sidx]; p<am.zone1_ptr[sidx+1]; ++p) z1sites.push_back(am.zone1_indices[p]);
        std::string zone1_combo;
        if (am.site_type == "top") {
            for (int site : z1sites) {
                int ai = code_to_activity_element(am, ce, occ[site]);
                if (ai >= 0) e += am.zone1_element[ai];
            }
        } else {
            zone1_combo = sorted_combo_key(am, ce, occ, z1sites);
            if (zone1_combo.empty()) return false;
            auto it = am.zone1_combo_coeff.find(zone1_combo);
            if (it == am.zone1_combo_coeff.end()) {
                for (int site : z1sites) {
                    int ai = code_to_activity_element(am, ce, occ[site]);
                    if (ai >= 0) e += am.zone1_element[ai];
                }
            } else {
                e += it->second;
            }
        }
        e += add_zone_element_terms(ce, am, occ, am.zone2_ptr, am.zone2_indices, am.zone2, sidx);
        if (!am.zone3_combo_coeff.empty() && !am.zone3_ptr.empty()) {
            std::vector<int> z3sites;
            for (int p=am.zone3_ptr[sidx]; p<am.zone3_ptr[sidx+1]; ++p) z3sites.push_back(am.zone3_indices[p]);
            std::string z3combo = sorted_combo_key(am, ce, occ, z3sites);
            auto z3it = am.zone3_combo_coeff.find(z3combo);
            if (z3it != am.zone3_combo_coeff.end()) e += z3it->second;
            else e += add_zone_element_terms(ce, am, occ, am.zone3_ptr, am.zone3_indices, am.zone3, sidx);
        } else {
            e += add_zone_element_terms(ce, am, occ, am.zone3_ptr, am.zone3_indices, am.zone3, sidx);
        }
        e += add_zone_element_terms(ce, am, occ, am.zone4_ptr, am.zone4_indices, am.zone4, sidx);
        e += add_zone_element_terms(ce, am, occ, am.zone5_ptr, am.zone5_indices, am.zone5, sidx);
        // Trial-level element-specific BE correction.  Top sites use the shift
        // of the single binding atom.  Fcc hollow sites use the average shift of
        // the binding atoms, keeping the correction on a per-site eV scale.
        double site_shift = 0.0;
        int n_shift = 0;
        for (int site : z1sites) {
            int ai = code_to_activity_element(am, ce, occ[site]);
            if (ai >= 0) { site_shift += be_shift[ai]; n_shift++; }
        }
        if (n_shift > 0) e -= site_shift / static_cast<double>(n_shift);
        if (zone1_combo_out) *zone1_combo_out = zone1_combo;
        return true;
}

static double predict_activity(const CEData &ce, const ActivityModel &am, const std::vector<int> &occ, const std::array<double,5> &be_shift) {
    // Original v1.0 activity: mean(exp(-abs(E_H-E_opt)/kBT)).
    int n = static_cast<int>(am.zone1_ptr.size()) - 1;
    if (n <= 0) return -std::numeric_limits<double>::infinity();
    double Tact = am.activity_temperature > 0.0 ? am.activity_temperature : 298.0;
    double kT = KB_EV_PER_K * Tact;
    double logsum = -std::numeric_limits<double>::infinity();
    int n_valid = 0;
    for (int sidx=0; sidx<n; ++sidx) {
        double e = 0.0;
        if (!binding_energy_for_site(ce, am, occ, be_shift, sidx, e)) continue;
        double ln_site = -std::abs(e - am.e_opt) / kT;
        if (!std::isfinite(logsum)) logsum = ln_site;
        else if (ln_site > logsum) logsum = ln_site + std::log1p(std::exp(logsum - ln_site));
        else logsum = logsum + std::log1p(std::exp(ln_site - logsum));
        n_valid++;
    }
    if (n_valid == 0) return 0.0;
    double log_mean = logsum - std::log(static_cast<double>(n_valid));
    if (log_mean < -745.0) return 0.0;
    return std::exp(log_mean);
}

static uint64_t composition_seed(const Config &cfg, int trial, int comp_idx) {
    return make_seed(cfg.random_seed, trial, comp_idx, 17, 0);
}

static uint64_t be_shift_seed(const Config &cfg, int trial) {
    // BE prediction uncertainty is a trial-level systematic model error.  The
    // five sampled values are applied according to the element(s) in the H
    // binding site: top uses one element; fcc hollow uses the average over the
    // binding atoms.
    return make_seed(cfg.random_seed, trial, 0, 202607, 505);
}

static std::array<double,5> sample_be_shifts(const Config &cfg, int trial) {
    std::mt19937_64 rng(be_shift_seed(cfg, trial));
    std::normal_distribution<double> be_norm(cfg.be_error_mean, cfg.be_error_sigma);
    std::array<double,5> shifts{};
    for (int k=0; k<5; ++k) shifts[k] = be_norm(rng);
    return shifts;
}

static uint64_t random_slab_seed(const Config &cfg, int trial, int comp_idx, int run_idx) {
    return make_seed(cfg.random_seed, trial, comp_idx, run_idx, 777);
}

static uint64_t cemc_seed(const Config &cfg, int trial, int comp_idx, int run_idx) {
    return make_seed(cfg.random_seed, trial, comp_idx, run_idx, 0);
}

struct SampledComposition {
    std::array<double,5> delta{};
    std::array<double,5> fraction{};
    std::array<int,5> counts{};
    std::array<double,5> be_shift{};
};

static SampledComposition sample_composition(const CompositionRow &row, const Config &cfg, int n_metal, int trial, int comp_idx) {
    std::mt19937_64 rng(composition_seed(cfg, trial, comp_idx));
    std::normal_distribution<double> comp_norm(cfg.comp_error_mean, cfg.comp_error_sigma);
    SampledComposition s;
    double unit_scale = (cfg.comp_error_units == "percent" || cfg.comp_error_units == "at_percent") ? 0.01 : 1.0;
    double total = 0.0;
    for (int k=0;k<5;k++) {
        s.delta[k] = comp_norm(rng) * unit_scale;
        s.fraction[k] = row.fraction[k] - s.delta[k];
        if (s.fraction[k] < 0.0) s.fraction[k] = 0.0;
        total += s.fraction[k];
    }
    if (total <= 1e-15) {
        s.fraction = row.fraction;
        total = std::accumulate(s.fraction.begin(), s.fraction.end(), 0.0);
    }
    for (double &x : s.fraction) x /= total;
    s.be_shift = sample_be_shifts(cfg, trial);

    std::array<double,5> exact{};
    std::array<double,5> fracpart{};
    int sum = 0;
    for (int k=0;k<5;k++) {
        exact[k] = s.fraction[k] * n_metal;
        s.counts[k] = static_cast<int>(std::floor(exact[k]));
        fracpart[k] = exact[k] - s.counts[k];
        sum += s.counts[k];
    }
    int remaining = n_metal - sum;
    std::array<int,5> order = {0,1,2,3,4};
    std::sort(order.begin(), order.end(), [&](int a, int b){ return fracpart[a] > fracpart[b]; });
    for (int i=0; i<remaining; ++i) s.counts[order[i % 5]]++;
    return s;
}

static std::vector<int> make_initial_occupations(const CEData &ce, const SampledComposition &s, const std::array<int,5> &element_to_code, std::mt19937_64 &rng) {
    std::vector<int> occ(ce.n_sites, ce.vacancy_code);
    std::vector<int> metal_codes;
    metal_codes.reserve(ce.n_metal_sites);
    for (int k=0;k<5;k++) {
        for (int c=0;c<s.counts[k];c++) metal_codes.push_back(element_to_code[k]);
    }
    if (static_cast<int>(metal_codes.size()) != ce.n_metal_sites) throw std::runtime_error("Counts do not sum to n_metal_sites");
    deterministic_shuffle(metal_codes, rng);
    for (int i=0; i<ce.n_metal_sites; ++i) occ[ce.metal_sites[i]] = metal_codes[i];
    return occ;
}

static std::vector<int> physical_atomic_numbers(const CEData &ce, const std::vector<int> &occ) {
    std::vector<int> z;
    z.reserve(ce.metal_sites.size());
    for (int site : ce.metal_sites) z.push_back(ce.atomic_numbers[occ[site]]);
    return z;
}

static void write_structure_json(std::ofstream &out, int trial, int comp_idx, const CompositionRow &row, int run_idx, int temp_idx, double target_temp,
                                 const std::string &method, double activity, const Snapshot *snap, const CEData &ce, const std::vector<int> &occ) {
    out << temp_idx << "\t" << comp_idx << "\t";
    out << "{\"trial\":" << trial
        << ",\"composition_index\":" << comp_idx
        << ",\"row_idx\":" << row.row_idx
        << ",\"col_idx\":" << row.col_idx
        << ",\"run\":" << run_idx
        << ",\"target_temperature\":" << std::fixed << std::setprecision(8) << target_temp
        << ",\"method\":\"" << method << "\""
        << ",\"activity\":" << std::setprecision(12) << activity;
    if (snap) {
        out << ",\"actual_temperature\":" << std::setprecision(8) << snap->actual_temperature
            << ",\"temperature_error\":" << std::setprecision(8) << snap->temperature_error
            << ",\"mc_step\":" << snap->global_step
            << ",\"energy\":" << std::setprecision(12) << snap->energy
            << ",\"attempted\":" << snap->attempted
            << ",\"accepted\":" << snap->accepted;
    }
    out << ",\"Z\":[";
    auto z = physical_atomic_numbers(ce, occ);
    for (size_t i=0;i<z.size();i++) {
        if (i) out << ',';
        out << z[i];
    }
    out << "]}\n";
}

static std::vector<double> scale_to_minus1_0(const std::vector<double> &v, bool better_is_higher) {
    std::vector<double> out(v.size(), 0.0);
    if (v.empty()) return out;
    auto [minit, maxit] = std::minmax_element(v.begin(), v.end());
    double minv = *minit, maxv = *maxit;
    double den = maxv - minv;
    if (std::abs(den) < 1e-30) return out;
    for (size_t i=0;i<v.size();i++) {
        if (better_is_higher) out[i] = -(v[i] - minv) / den;     // max -> -1, min -> 0
        else out[i] = -(maxv - v[i]) / den;                      // min -> -1, max -> 0
    }
    return out;
}

static double mse(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size() || a.empty()) return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    for (size_t i=0;i<a.size();i++) { double d = a[i] - b[i]; s += d*d; }
    return s / static_cast<double>(a.size());
}

static int signum(double x) { return (x > 0) - (x < 0); }

static double kendall_tau_b(const std::vector<double> &x, const std::vector<double> &y) {
    if (x.size() != y.size() || x.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    long double concord = 0, discord = 0, ties_x = 0, ties_y = 0;
    const double eps = 1e-14;
    for (size_t i=0;i<x.size();i++) {
        for (size_t j=i+1;j<x.size();j++) {
            int sx = (std::abs(x[i]-x[j]) <= eps) ? 0 : signum(x[i]-x[j]);
            int sy = (std::abs(y[i]-y[j]) <= eps) ? 0 : signum(y[i]-y[j]);
            if (sx == 0 && sy == 0) continue;
            if (sx == 0) ties_x += 1;
            else if (sy == 0) ties_y += 1;
            else if (sx == sy) concord += 1;
            else discord += 1;
        }
    }
    long double den = std::sqrt((concord + discord + ties_x) * (concord + discord + ties_y));
    if (den <= 0) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>((concord - discord) / den);
}

static std::set<int> read_completed_trials(const fs::path &outdir) {
    std::set<int> done;
    std::ifstream in(outdir / "completed_trials.txt");
    int t;
    while (in >> t) done.insert(t);
    return done;
}

static void filter_file_remove_trial(const fs::path &path, int trial, bool csv) {
    if (!fs::exists(path)) return;
    fs::path tmp = path;
    tmp += ".repair";
    std::ifstream in(path);
    std::ofstream out(tmp);
    std::string line;
    bool first = true;
    const std::string needle = "\"trial\":" + std::to_string(trial);
    while (std::getline(in, line)) {
        bool keep = true;
        if (csv) {
            if (!first && starts_with(line, std::to_string(trial) + ",")) keep = false;
        } else {
            if (line.find(needle) != std::string::npos) keep = false;
        }
        if (keep) out << line << "\n";
        first = false;
    }
    in.close(); out.close();
    fs::rename(tmp, path);
}

static void filter_completed_remove_trial(const fs::path &path, int trial) {
    if (!fs::exists(path)) return;
    fs::path tmp = path;
    tmp += ".repair";
    std::ifstream in(path);
    std::ofstream out(tmp);
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        try {
            if (std::stoi(trim(line)) == trial) continue;
        } catch (...) {}
        out << line << "\n";
    }
    in.close(); out.close();
    fs::rename(tmp, path);
}

static fs::path structure_file_path(const fs::path &outdir, double target_temperature, int comp_idx) {
    int T = static_cast<int>(std::llround(target_temperature));
    std::ostringstream tdir;
    tdir << "T" << std::setw(5) << std::setfill('0') << T;
    std::ostringstream cname;
    cname << "comp_" << std::setw(4) << std::setfill('0') << comp_idx << ".jsonl";
    return outdir / "structures_by_temperature" / tdir.str() / cname.str();
}

static void repair_partial_merge(const fs::path &outdir, const SchedulePlan &plan) {
    fs::path marker = outdir / "merge_in_progress_trial.txt";
    if (!fs::exists(marker)) return;
    std::ifstream in(marker);
    int trial = -1;
    in >> trial;
    if (trial < 0) return;
    std::cerr << "Repairing partial merge for trial " << trial << "...\n";
    filter_file_remove_trial(outdir / "metrics_by_trial_temperature.csv", trial, true);
    filter_file_remove_trial(outdir / "random_metrics_by_trial.csv", trial, true);
    filter_file_remove_trial(outdir / "selected_temperature_by_trial.csv", trial, true);
    filter_file_remove_trial(outdir / "paired_comparison_by_trial.csv", trial, true);
    filter_file_remove_trial(outdir / "trial_composition_samples.csv", trial, true);
    filter_file_remove_trial(outdir / "random_slab_seeds.csv", trial, true);
    filter_file_remove_trial(outdir / "predicted_activity_selected_by_trial.csv", trial, true);
    filter_completed_remove_trial(outdir / "completed_trials.txt", trial);
    fs::remove(outdir / "uncertainty_summary.csv");
    fs::remove(outdir / "method_comparison_ci.csv");
    fs::remove(outdir / "temperature_selection_counts.csv");
    fs::remove(outdir / "uncertainty_decision.csv");
    fs::remove(outdir / "uncertainty_comparison_summary.csv");
    fs::remove(outdir / "plot_trial_metrics_long.csv");
    fs::remove(outdir / "plot_tau_mse_scatter.csv");
    fs::remove(outdir / "plot_paired_deltas_long.csv");
    fs::remove(outdir / "plot_metric_summary.csv");
    fs::remove(outdir / "plot_method_comparison_ci.csv");
    fs::remove(outdir / "plot_cemc_temperature_metrics_long.csv");
    fs::remove(outdir / "plot_cemc_temperature_summary.csv");
    fs::remove(outdir / "plot_temperature_selection_counts.csv");
    fs::remove_all(outdir / "plot_data");
    fs::remove(outdir / "plot_tau_mse_scatter.csv");
    fs::remove(outdir / "plot_metric_by_trial_long.csv");
    fs::remove(outdir / "plot_paired_deltas.csv");
    fs::remove(outdir / "plot_ci_intervals.csv");
    fs::remove(outdir / "plot_temperature_response_long.csv");
    fs::remove(outdir / "plot_temperature_response_summary.csv");
    fs::remove(outdir / "plot_manifest.txt");
    fs::remove_all(outdir / "plot_ready");

    std::string pstr;
    std::set<fs::path> touched;
    while (in >> pstr) touched.insert(fs::path(pstr));
    if (touched.empty()) {
        // Backward-compatible repair for older temperature-wide structure files.
        for (size_t ti=0; ti<plan.target_temperatures.size(); ++ti) {
            int T = static_cast<int>(std::llround(plan.target_temperatures[ti]));
            std::ostringstream name;
            name << "structures_T" << std::setw(5) << std::setfill('0') << T << ".jsonl";
            touched.insert(outdir / name.str());
        }
        fs::path root = outdir / "structures_by_temperature";
        if (fs::exists(root)) {
            for (auto const &entry : fs::recursive_directory_iterator(root)) {
                if (entry.is_regular_file() && entry.path().extension() == ".jsonl") touched.insert(entry.path());
            }
        }
    }
    for (const auto &p : touched) filter_file_remove_trial(p, trial, false);
    fs::remove(marker);
}

static void ensure_headers(const fs::path &outdir) {
    fs::create_directories(outdir);
    // v1.0 compact predicted-activity mode: keep only the selected-temperature
    // composition-level file.  The all-temperature CEMC and random-only files
    // from older v1.0 drafts are intentionally not produced.
    fs::remove(outdir / "predicted_activity_cemc_by_trial_temperature.csv");
    fs::remove(outdir / "predicted_activity_random_by_trial.csv");
    auto create_with_header = [](const fs::path &p, const std::string &h) {
        if (!fs::exists(p) || fs::file_size(p) == 0) {
            std::ofstream out(p);
            out << h << "\n";
            return;
        }
        std::ifstream in(p);
        std::string first;
        std::getline(in, first);
        if (trim(first) != h) {
            throw std::runtime_error(
                "Existing output file has an old/incompatible header: " + p.string() +
                "\nUse a new workdir/output_dir, or move/remove the old results directory before running v1.0."
            );
        }
    };
    create_with_header(outdir / "metrics_by_trial_temperature.csv",
        "trial,temp_idx,temperature,method,tau,mse,score_tau_minus_mse,n_compositions,n_run_records");
    create_with_header(outdir / "random_metrics_by_trial.csv",
        "trial,method,tau,mse,score_tau_minus_mse,n_compositions,n_run_records");
    create_with_header(outdir / "selected_temperature_by_trial.csv",
        "trial,cemc_selected_temp,cemc_score_tau_minus_mse,cemc_tau,cemc_mse,random_score_tau_minus_mse,random_tau,random_mse,delta_tau_cemc_minus_random,delta_mse_random_minus_cemc,delta_score_cemc_minus_random");
    create_with_header(outdir / "paired_comparison_by_trial.csv",
        "trial,cemc_selected_temp,cemc_selected_tau,cemc_selected_mse,cemc_selected_score,random_tau,random_mse,random_score,delta_tau_cemc_minus_random,delta_mse_random_minus_cemc,delta_score_cemc_minus_random");
    create_with_header(outdir / "trial_composition_samples.csv",
        "trial,composition_index,row_idx,col_idx,orig_Fe,orig_Co,orig_Ni,orig_Pd,orig_Pt,delta_Fe,delta_Co,delta_Ni,delta_Pd,delta_Pt,sampled_Fe,sampled_Co,sampled_Ni,sampled_Pd,sampled_Pt,count_Fe,count_Co,count_Ni,count_Pd,count_Pt,be_shift_Fe,be_shift_Co,be_shift_Ni,be_shift_Pd,be_shift_Pt");
    create_with_header(outdir / "random_slab_seeds.csv",
        "trial,composition_index,row_idx,col_idx,run,composition_sample_seed,random_slab_seed,cemc_seed,base_random_seed,seed_role,shuffle_algorithm,n_metal_sites,count_Fe,count_Co,count_Ni,count_Pd,count_Pt,sampled_Fe,sampled_Co,sampled_Ni,sampled_Pd,sampled_Pt,be_shift_Fe,be_shift_Co,be_shift_Ni,be_shift_Pd,be_shift_Pt");
    create_with_header(outdir / "predicted_activity_selected_by_trial.csv",
        "trial,composition_index,row_idx,col_idx,selected_temp_idx,selected_temperature,cemc_pred_activity_mean,cemc_pred_activity_sd,cemc_n_runs,random_pred_activity_mean,random_pred_activity_sd,random_n_runs,experimental_activity,cemc_pred_activity_scaled,random_pred_activity_scaled,experimental_activity_scaled");
    create_with_header(outdir / "trial_timing.csv",
        "trial,compute_seconds,merge_seconds,summary_seconds,total_seconds,n_tasks,mpi_ranks,structures_written,summary_written");
    create_with_header(outdir / "uncertainty_summary.csv",
        "quantity,metric,n_trials,mean,sd,sem,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion");
    create_with_header(outdir / "method_comparison_ci.csv",
        "comparison,metric,n_trials,mean_delta,sd_delta,sem_delta,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion,mean_ci_supports_cemc_better,empirical95_supports_cemc_better");
    create_with_header(outdir / "temperature_selection_counts.csv",
        "temperature,n_selected_trials,fraction_selected");
}

struct ResultRecord {
    int trial=0, comp=0, run=0, temp_idx=0;
    double temp=0.0;
    std::string method;
    double activity=0.0;
    double energy=0.0;
    int64_t attempted=0, accepted=0;
};

static ResultRecord parse_result_line(const std::string &line) {
    auto v = split_csv_line(line);
    if (v.size() < 10) throw std::runtime_error("Bad result shard line: " + line);
    ResultRecord r;
    r.trial = std::stoi(v[0]);
    r.comp = std::stoi(v[1]);
    r.run = std::stoi(v[2]);
    r.temp_idx = std::stoi(v[3]);
    r.temp = std::stod(v[4]);
    r.method = v[5];
    r.activity = std::stod(v[6]);
    r.energy = std::stod(v[7]);
    r.attempted = static_cast<int64_t>(std::stoll(v[8]));
    r.accepted = static_cast<int64_t>(std::stoll(v[9]));
    return r;
}

static void append_trial_samples(const fs::path &outdir, int trial, const std::vector<CompositionRow> &rows, const Config &cfg, int n_metal) {
    std::ofstream out(outdir / "trial_composition_samples.csv", std::ios::app);
    out << std::setprecision(12);
    for (int ci=0; ci<(int)rows.size(); ++ci) {
        auto s = sample_composition(rows[ci], cfg, n_metal, trial, ci);
        out << trial << ',' << ci << ',' << rows[ci].row_idx << ',' << rows[ci].col_idx;
        for (double x : rows[ci].fraction) out << ',' << x;
        for (double x : s.delta) out << ',' << x;
        for (double x : s.fraction) out << ',' << x;
        for (int x : s.counts) out << ',' << x;
        for (double x : s.be_shift) out << ',' << x;
        out << "\n";
    }
}

static void append_random_seed_shards(const fs::path &outdir, const fs::path &trial_dir, int world_size) {
    std::ofstream out(outdir / "random_slab_seeds.csv", std::ios::app);
    for (int rank=0; rank<world_size; ++rank) {
        std::ostringstream rp;
        rp << "rank_" << std::setw(5) << std::setfill('0') << rank << "_seeds.csv";
        fs::path p = trial_dir / rp.str();
        if (!fs::exists(p)) continue;
        std::ifstream in(p);
        std::string line;
        while (std::getline(in, line)) {
            if (!trim(line).empty()) out << line << "\n";
        }
    }
}

struct RandomSeedRecord {
    int trial = -1;
    int comp = -1;
    int row_idx = -1;
    int col_idx = -1;
    int run = -1;
    uint64_t random_slab_seed = 0;
    std::array<int,5> counts{};
};

static std::unordered_map<std::string, int> csv_header_map(const std::string &header) {
    auto names = split_csv_line(header);
    std::unordered_map<std::string, int> out;
    for (int i=0; i<(int)names.size(); ++i) out[names[i]] = i;
    return out;
}

static int required_col(const std::unordered_map<std::string,int> &m, const std::string &name) {
    auto it = m.find(name);
    if (it == m.end()) throw std::runtime_error("random_slab_seeds.csv missing column: " + name);
    return it->second;
}

static RandomSeedRecord find_random_seed_record(const fs::path &outdir, int trial, int comp, int run) {
    fs::path p = outdir / "random_slab_seeds.csv";
    std::ifstream in(p);
    if (!in) throw std::runtime_error("Could not open random slab seed file: " + p.string());
    std::string header;
    if (!std::getline(in, header)) throw std::runtime_error("Empty random_slab_seeds.csv");
    auto h = csv_header_map(header);
    const int c_trial = required_col(h, "trial");
    const int c_comp = required_col(h, "composition_index");
    const int c_run = required_col(h, "run");
    const int c_row = required_col(h, "row_idx");
    const int c_col = required_col(h, "col_idx");
    const int c_rseed = required_col(h, "random_slab_seed");
    const std::array<std::string,5> count_cols = {"count_Fe","count_Co","count_Ni","count_Pd","count_Pt"};
    std::array<int,5> c_count{};
    for (int k=0; k<5; ++k) c_count[k] = required_col(h, count_cols[k]);
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        auto v = split_csv_line(line);
        if ((int)v.size() < (int)h.size()) throw std::runtime_error("Bad random_slab_seeds.csv line: " + line);
        int t = std::stoi(v[c_trial]);
        int c = std::stoi(v[c_comp]);
        int r = std::stoi(v[c_run]);
        if (t == trial && c == comp && r == run) {
            RandomSeedRecord rec;
            rec.trial = t;
            rec.comp = c;
            rec.row_idx = std::stoi(v[c_row]);
            rec.col_idx = std::stoi(v[c_col]);
            rec.run = r;
            rec.random_slab_seed = static_cast<uint64_t>(std::stoull(v[c_rseed]));
            for (int k=0; k<5; ++k) rec.counts[k] = std::stoi(v[c_count[k]]);
            return rec;
        }
    }
    throw std::runtime_error("No matching random slab seed row for trial=" + std::to_string(trial) +
                             ", composition=" + std::to_string(comp) + ", run=" + std::to_string(run));
}

static std::array<int,5> parse_counts_arg(const std::string &text) {
    std::array<int,5> counts{};
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : text) {
        if (ch == ',' || ch == ':' || std::isspace(static_cast<unsigned char>(ch))) {
            if (!cur.empty()) { parts.push_back(cur); cur.clear(); }
        } else cur.push_back(ch);
    }
    if (!cur.empty()) parts.push_back(cur);
    if (parts.size() != 5) throw std::runtime_error("--counts must contain five integers: Fe,Co,Ni,Pd,Pt");
    for (int k=0; k<5; ++k) counts[k] = std::stoi(parts[k]);
    return counts;
}

static void write_atomic_numbers_json(const fs::path &outpath, const CEData &ce, const std::vector<int> &occ) {
    if (!outpath.parent_path().empty()) fs::create_directories(outpath.parent_path());
    std::ofstream out(outpath);
    if (!out) throw std::runtime_error("Could not write random slab JSON: " + outpath.string());
    auto z = physical_atomic_numbers(ce, occ);
    out << "[";
    for (size_t i=0; i<z.size(); ++i) { if (i) out << ","; out << z[i]; }
    out << "]\n";
}

static void reconstruct_random_slab_file(const CEData &ce, const std::array<int,5> &element_to_code,
                                         uint64_t seed, const std::array<int,5> &counts,
                                         const fs::path &output_path) {
    SampledComposition s;
    s.counts = counts;
    std::mt19937_64 rng(seed);
    auto occ = make_initial_occupations(ce, s, element_to_code, rng);
    write_atomic_numbers_json(output_path, ce, occ);
}



struct Metric { double tau = std::numeric_limits<double>::quiet_NaN(); double mse = std::numeric_limits<double>::quiet_NaN(); };
static double metric_score(const Metric &m) { return m.tau - m.mse; }

struct SimpleStats {
    int n = 0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double stddev = std::numeric_limits<double>::quiet_NaN();
    double sem = std::numeric_limits<double>::quiet_NaN();
    double ci_low = std::numeric_limits<double>::quiet_NaN();
    double ci_high = std::numeric_limits<double>::quiet_NaN();
    double q025 = std::numeric_limits<double>::quiet_NaN();
    double q50 = std::numeric_limits<double>::quiet_NaN();
    double q975 = std::numeric_limits<double>::quiet_NaN();
};

static double quantile_sorted(const std::vector<double> &v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (v.size() == 1) return v[0];
    double pos = q * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double t = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - t) + v[hi] * t;
}

static SimpleStats compute_stats(std::vector<double> values) {
    std::vector<double> v;
    for (double x : values) if (std::isfinite(x)) v.push_back(x);
    std::sort(v.begin(), v.end());
    SimpleStats s;
    s.n = static_cast<int>(v.size());
    if (v.empty()) return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.q025 = quantile_sorted(v, 0.025);
    s.q50 = quantile_sorted(v, 0.500);
    s.q975 = quantile_sorted(v, 0.975);
    if (v.size() >= 2) {
        double ss = 0.0;
        for (double x : v) { double d = x - s.mean; ss += d*d; }
        s.stddev = std::sqrt(ss / static_cast<double>(v.size() - 1));
        s.sem = s.stddev / std::sqrt(static_cast<double>(v.size()));
        s.ci_low = s.mean - 1.96 * s.sem;
        s.ci_high = s.mean + 1.96 * s.sem;
    } else {
        s.stddev = 0.0;
        s.sem = 0.0;
        s.ci_low = s.mean;
        s.ci_high = s.mean;
    }
    return s;
}

static void write_dist_row(std::ofstream &out, const std::string &quantity, const std::string &metric, const SimpleStats &s, const std::string &criterion) {
    out << std::setprecision(12)
        << quantity << ',' << metric << ',' << s.n << ',' << s.mean << ',' << s.stddev << ',' << s.sem << ','
        << s.ci_low << ',' << s.ci_high << ',' << s.q025 << ',' << s.q50 << ',' << s.q975 << ','
        << '"' << criterion << '"' << "\n";
}

static void write_comparison_row(std::ofstream &out, const std::string &comparison, const std::string &metric, const SimpleStats &s, const std::string &criterion) {
    bool mean_support = std::isfinite(s.ci_low) && s.ci_low > 0.0;
    bool empirical_support = std::isfinite(s.q025) && s.q025 > 0.0;
    out << std::setprecision(12)
        << comparison << ',' << metric << ',' << s.n << ',' << s.mean << ',' << s.stddev << ',' << s.sem << ','
        << s.ci_low << ',' << s.ci_high << ',' << s.q025 << ',' << s.q50 << ',' << s.q975 << ','
        << '"' << criterion << '"' << ','
        << (mean_support ? "true" : "false") << ',' << (empirical_support ? "true" : "false") << "\n";
}


static void write_uncertainty_summary(const fs::path &outdir, int ml_index) {
    fs::path selected_path = outdir / "selected_temperature_by_trial.csv";
    if (!fs::exists(selected_path)) return;
    std::ifstream in(selected_path);
    std::string line;
    std::getline(in, line); // header

    struct SelectedRow {
        int trial = 0;
        double temp = std::numeric_limits<double>::quiet_NaN();
        double cemc_score = std::numeric_limits<double>::quiet_NaN();
        double cemc_tau = std::numeric_limits<double>::quiet_NaN();
        double cemc_mse = std::numeric_limits<double>::quiet_NaN();
        double random_score = std::numeric_limits<double>::quiet_NaN();
        double random_tau = std::numeric_limits<double>::quiet_NaN();
        double random_mse = std::numeric_limits<double>::quiet_NaN();
        double delta_tau = std::numeric_limits<double>::quiet_NaN();      // CEMC tau - random tau; positive is better for CEMC
        double delta_mse = std::numeric_limits<double>::quiet_NaN();      // random MSE - CEMC MSE; positive is better for CEMC
        double delta_score = std::numeric_limits<double>::quiet_NaN();    // CEMC score - random score; positive is better for CEMC
    };

    std::vector<SelectedRow> selected_rows;
    std::vector<double> cemc_temp, cemc_score, cemc_tau, cemc_mse, random_score, random_tau, random_mse, delta_tau, delta_mse, delta_score;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        auto v = split_csv_line(line);
        if (v.size() < 11) continue;
        try {
            SelectedRow r;
            r.trial = std::stoi(v[0]);
            r.temp = std::stod(v[1]);
            r.cemc_score = std::stod(v[2]);
            r.cemc_tau = std::stod(v[3]);
            r.cemc_mse = std::stod(v[4]);
            r.random_score = std::stod(v[5]);
            r.random_tau = std::stod(v[6]);
            r.random_mse = std::stod(v[7]);
            r.delta_tau = std::stod(v[8]);
            r.delta_mse = std::stod(v[9]);
            r.delta_score = std::stod(v[10]);
            selected_rows.push_back(r);
            cemc_temp.push_back(r.temp);
            cemc_score.push_back(r.cemc_score);
            cemc_tau.push_back(r.cemc_tau);
            cemc_mse.push_back(r.cemc_mse);
            random_score.push_back(r.random_score);
            random_tau.push_back(r.random_tau);
            random_mse.push_back(r.random_mse);
            delta_tau.push_back(r.delta_tau);
            // Positive delta_mse means CEMC MSE is lower because it is random_MSE - CEMC_MSE.
            delta_mse.push_back(r.delta_mse);
            delta_score.push_back(r.delta_score);
        } catch (...) {}
    }
    if (selected_rows.empty()) return;

    auto st_cemc_tau = compute_stats(cemc_tau);
    auto st_random_tau = compute_stats(random_tau);
    auto st_delta_tau = compute_stats(delta_tau);
    auto st_cemc_mse = compute_stats(cemc_mse);
    auto st_random_mse = compute_stats(random_mse);
    auto st_delta_mse = compute_stats(delta_mse);
    auto st_cemc_score = compute_stats(cemc_score);
    auto st_random_score = compute_stats(random_score);
    auto st_delta_score = compute_stats(delta_score);
    auto st_temp = compute_stats(cemc_temp);

    // Legacy compact summaries.
    {
        std::ofstream out(outdir / "uncertainty_summary.csv");
        out << "quantity,metric,n_trials,mean,sd,sem,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion\n";
        write_dist_row(out, "cemc_selected", "tau", st_cemc_tau, "higher tau is better");
        write_dist_row(out, "random", "tau", st_random_tau, "higher tau is better; random has no temperature");
        write_dist_row(out, "cemc_minus_random", "tau", st_delta_tau, "positive means CEMC tau is higher");
        write_dist_row(out, "cemc_selected", "mse", st_cemc_mse, "lower MSE is better");
        write_dist_row(out, "random", "mse", st_random_mse, "lower MSE is better; random has no temperature");
        write_dist_row(out, "random_minus_cemc", "mse", st_delta_mse, "positive means CEMC MSE is lower");
        write_dist_row(out, "cemc_selected", "score_tau_minus_mse", st_cemc_score, "temperature is selected by maximizing tau - MSE");
        write_dist_row(out, "random", "score_tau_minus_mse", st_random_score, "random tau - MSE; no temperature selection");
        write_dist_row(out, "cemc_minus_random", "score_tau_minus_mse", st_delta_score, "positive means selected CEMC score is higher");
        write_dist_row(out, "cemc_selected", "temperature", st_temp, "distribution of selected MC slab temperatures across UQ trials");
    }
    {
        std::ofstream out(outdir / "method_comparison_ci.csv");
        out << "comparison,metric,n_trials,mean_delta,sd_delta,sem_delta,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion,mean_ci_supports_cemc_better,empirical95_supports_cemc_better\n";
        write_comparison_row(out, "cemc_tau_minus_random_tau", "tau", st_delta_tau, "positive means CEMC tau is higher");
        write_comparison_row(out, "random_mse_minus_cemc_mse", "mse", st_delta_mse, "positive means CEMC MSE is lower");
        write_comparison_row(out, "cemc_score_minus_random_score", "score_tau_minus_mse", st_delta_score, "positive means selected CEMC tau-MSE score is higher");
    }

    // Plot-ready long files. These are regenerated from selected_temperature_by_trial.csv.
    {
        std::ofstream out(outdir / "plot_trial_metrics_long.csv");
        out << "ml,trial,method,selected_temperature,tau,mse,score_tau_minus_mse\n";
        out << std::setprecision(12);
        for (const auto &r : selected_rows) {
            out << ml_index << ',' << r.trial << ",cemc_selected," << r.temp << ',' << r.cemc_tau << ',' << r.cemc_mse << ',' << r.cemc_score << "\n";
            out << ml_index << ',' << r.trial << ",random,," << r.random_tau << ',' << r.random_mse << ',' << r.random_score << "\n";
        }
    }
    {
        std::ofstream out(outdir / "plot_tau_mse_scatter.csv");
        out << "ml,trial,method,selected_temperature,tau,mse,score_tau_minus_mse\n";
        out << std::setprecision(12);
        for (const auto &r : selected_rows) {
            out << ml_index << ',' << r.trial << ",cemc_selected," << r.temp << ',' << r.cemc_tau << ',' << r.cemc_mse << ',' << r.cemc_score << "\n";
            out << ml_index << ',' << r.trial << ",random,," << r.random_tau << ',' << r.random_mse << ',' << r.random_score << "\n";
        }
    }
    {
        std::ofstream out(outdir / "plot_paired_deltas_long.csv");
        out << "ml,trial,metric,delta,positive_means,cemc_value,random_value\n";
        out << std::setprecision(12);
        for (const auto &r : selected_rows) {
            out << ml_index << ',' << r.trial << ",tau," << r.delta_tau << ",CEMC higher tau," << r.cemc_tau << ',' << r.random_tau << "\n";
            out << ml_index << ',' << r.trial << ",mse," << r.delta_mse << ",CEMC lower MSE," << r.cemc_mse << ',' << r.random_mse << "\n";
            out << ml_index << ',' << r.trial << ",score_tau_minus_mse," << r.delta_score << ",CEMC higher tau-MSE score," << r.cemc_score << ',' << r.random_score << "\n";
        }
    }

    auto write_plot_summary = [&](std::ofstream &out, const std::string &quantity, const std::string &metric, const SimpleStats &s, const std::string &criterion) {
        out << std::setprecision(12)
            << ml_index << ',' << quantity << ',' << metric << ',' << s.n << ',' << s.mean << ',' << s.stddev << ',' << s.sem << ','
            << s.ci_low << ',' << s.ci_high << ',' << s.q025 << ',' << s.q50 << ',' << s.q975 << ',' << '"' << criterion << '"' << "\n";
    };
    {
        std::ofstream out(outdir / "plot_metric_summary.csv");
        out << "ml,quantity,metric,n_trials,mean,sd,sem,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion\n";
        write_plot_summary(out, "cemc_selected", "tau", st_cemc_tau, "higher tau is better");
        write_plot_summary(out, "random", "tau", st_random_tau, "higher tau is better; random has no temperature");
        write_plot_summary(out, "cemc_selected", "mse", st_cemc_mse, "lower MSE is better");
        write_plot_summary(out, "random", "mse", st_random_mse, "lower MSE is better; random has no temperature");
        write_plot_summary(out, "cemc_minus_random", "tau", st_delta_tau, "positive means CEMC tau is higher");
        write_plot_summary(out, "random_minus_cemc", "mse", st_delta_mse, "positive means CEMC MSE is lower");
        write_plot_summary(out, "cemc_minus_random", "score_tau_minus_mse", st_delta_score, "positive means selected CEMC tau-MSE score is higher");
        write_plot_summary(out, "cemc_selected", "temperature", st_temp, "selected MC slab temperature distribution");
    }
    {
        std::ofstream out(outdir / "plot_method_comparison_ci.csv");
        out << "ml,comparison,metric,n_trials,mean_delta,sd_delta,sem_delta,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion,mean_ci_supports_cemc_better,empirical95_supports_cemc_better\n";
        auto wr = [&](const std::string &comparison, const std::string &metric, const SimpleStats &s, const std::string &criterion) {
            bool mean_support = std::isfinite(s.ci_low) && s.ci_low > 0.0;
            bool empirical_support = std::isfinite(s.q025) && s.q025 > 0.0;
            out << std::setprecision(12)
                << ml_index << ',' << comparison << ',' << metric << ',' << s.n << ',' << s.mean << ',' << s.stddev << ',' << s.sem << ','
                << s.ci_low << ',' << s.ci_high << ',' << s.q025 << ',' << s.q50 << ',' << s.q975 << ',' << '"' << criterion << '"' << ','
                << (mean_support ? "true" : "false") << ',' << (empirical_support ? "true" : "false") << "\n";
        };
        wr("cemc_tau_minus_random_tau", "tau", st_delta_tau, "positive means CEMC tau is higher");
        wr("random_mse_minus_cemc_mse", "mse", st_delta_mse, "positive means CEMC MSE is lower");
        wr("cemc_score_minus_random_score", "score_tau_minus_mse", st_delta_score, "positive means selected CEMC tau-MSE score is higher");
    }

    // Temperature-selection histogram data.
    std::map<int,int> counts;
    for (double t : cemc_temp) if (std::isfinite(t)) counts[static_cast<int>(std::llround(t))]++;
    int n = static_cast<int>(cemc_temp.size());
    {
        std::ofstream out(outdir / "temperature_selection_counts.csv");
        out << "temperature,n_selected_trials,fraction_selected\n";
        out << std::setprecision(12);
        for (const auto &kv : counts) out << kv.first << ',' << kv.second << ',' << (n > 0 ? static_cast<double>(kv.second)/n : 0.0) << "\n";
    }
    {
        std::ofstream out(outdir / "plot_temperature_selection_counts.csv");
        out << "ml,temperature,n_selected_trials,fraction_selected\n";
        out << std::setprecision(12);
        for (const auto &kv : counts) out << ml_index << ',' << kv.first << ',' << kv.second << ',' << (n > 0 ? static_cast<double>(kv.second)/n : 0.0) << "\n";
    }

    // CEMC metric at every saved temperature, for temperature-response line plots.
    fs::path metrics_path = outdir / "metrics_by_trial_temperature.csv";
    if (fs::exists(metrics_path)) {
        struct TempMetricRow { int trial; double temp; double tau; double mse; double score; };
        std::vector<TempMetricRow> temp_rows;
        std::ifstream mt(metrics_path);
        std::string mline;
        bool first = true;
        while (std::getline(mt, mline)) {
            if (trim(mline).empty()) continue;
            if (first) { first = false; continue; }
            auto v = split_csv_line(mline);
            if (v.size() < 9) continue;
            try {
                if (v[3] != "cemc") continue;
                TempMetricRow r;
                r.trial = std::stoi(v[0]);
                r.temp = std::stod(v[2]);
                r.tau = std::stod(v[4]);
                r.mse = std::stod(v[5]);
                r.score = std::stod(v[6]);
                temp_rows.push_back(r);
            } catch (...) {}
        }
        {
            std::ofstream out(outdir / "plot_cemc_temperature_metrics_long.csv");
            out << "ml,trial,temperature,metric,value\n";
            out << std::setprecision(12);
            for (const auto &r : temp_rows) {
                out << ml_index << ',' << r.trial << ',' << r.temp << ",tau," << r.tau << "\n";
                out << ml_index << ',' << r.trial << ',' << r.temp << ",mse," << r.mse << "\n";
                out << ml_index << ',' << r.trial << ',' << r.temp << ",score_tau_minus_mse," << r.score << "\n";
            }
        }
        std::map<int, std::vector<double>> byT_tau, byT_mse, byT_score;
        for (const auto &r : temp_rows) {
            int T = static_cast<int>(std::llround(r.temp));
            byT_tau[T].push_back(r.tau);
            byT_mse[T].push_back(r.mse);
            byT_score[T].push_back(r.score);
        }
        std::ofstream out(outdir / "plot_cemc_temperature_summary.csv");
        out << "ml,temperature,metric,n_trials,mean,sd,sem,ci95_mean_low,ci95_mean_high,q2p5,q50,q97p5,criterion\n";
        auto wrT = [&](int T, const std::string &metric, const std::vector<double> &vals, const std::string &criterion) {
            auto s = compute_stats(vals);
            out << std::setprecision(12)
                << ml_index << ',' << T << ',' << metric << ',' << s.n << ',' << s.mean << ',' << s.stddev << ',' << s.sem << ','
                << s.ci_low << ',' << s.ci_high << ',' << s.q025 << ',' << s.q50 << ',' << s.q975 << ',' << '"' << criterion << '"' << "\n";
        };
        for (const auto &kv : byT_tau) wrT(kv.first, "tau", kv.second, "higher tau is better");
        for (const auto &kv : byT_mse) wrT(kv.first, "mse", kv.second, "lower MSE is better");
        for (const auto &kv : byT_score) wrT(kv.first, "score_tau_minus_mse", kv.second, "higher tau-MSE score is better");
    }

    // Also mirror plot-ready files into results/plot_data/ with stable, graph-oriented names.
    // The source plot_*.csv files are kept at the root for backward compatibility.
    {
        fs::path plot_dir = outdir / "plot_data";
        fs::create_directories(plot_dir);
        auto cp = [&](const std::string &src_name, const std::string &dst_name) {
            fs::path src = outdir / src_name;
            if (fs::exists(src)) fs::copy_file(src, plot_dir / dst_name, fs::copy_options::overwrite_existing);
        };
        cp("plot_trial_metrics_long.csv", "trial_metric_distributions.csv");
        cp("plot_trial_metrics_long.csv", "method_metric_long_by_trial.csv");
        cp("plot_tau_mse_scatter.csv", "tau_mse_scatter.csv");
        cp("plot_tau_mse_scatter.csv", "tau_mse_scatter_by_trial.csv");
        cp("plot_paired_deltas_long.csv", "paired_deltas.csv");
        cp("plot_paired_deltas_long.csv", "paired_delta_long_by_trial.csv");
        cp("plot_metric_summary.csv", "ci_interval_summary.csv");
        cp("plot_metric_summary.csv", "method_metric_interval_summary.csv");
        cp("plot_method_comparison_ci.csv", "paired_delta_interval_summary.csv");
        cp("plot_method_comparison_ci.csv", "method_comparison_ci.csv");
        cp("plot_temperature_selection_counts.csv", "temperature_selection_counts.csv");
        cp("plot_cemc_temperature_metrics_long.csv", "temperature_metric_long.csv");
        cp("plot_cemc_temperature_metrics_long.csv", "cemc_temperature_metric_by_trial.csv");
        cp("plot_cemc_temperature_summary.csv", "temperature_metric_summary.csv");
        cp("plot_cemc_temperature_summary.csv", "cemc_temperature_summary.csv");

        // Wide paired-delta table for simple histogram plotting.
        std::ofstream wide(plot_dir / "paired_delta_by_trial.csv");
        wide << "ml,trial,cemc_selected_temperature,delta_tau_cemc_minus_random,delta_mse_random_minus_cemc,delta_score_cemc_minus_random,cemc_tau,random_tau,cemc_mse,random_mse,cemc_score,random_score\n";
        wide << std::setprecision(12);
        for (const auto &r : selected_rows) {
            wide << ml_index << ',' << r.trial << ',' << r.temp << ',' << r.delta_tau << ',' << r.delta_mse << ',' << r.delta_score << ','
                 << r.cemc_tau << ',' << r.random_tau << ',' << r.cemc_mse << ',' << r.random_mse << ',' << r.cemc_score << ',' << r.random_score << "\n";
        }
        wide.close();

        // Short-name aliases in results/plot_ready/.
        fs::path plot_ready = outdir / "plot_ready";
        fs::create_directories(plot_ready);
        auto cp_ready = [&](const std::string &src_name, const std::string &dst_name) {
            fs::path src = plot_dir / src_name;
            if (fs::exists(src)) fs::copy_file(src, plot_ready / dst_name, fs::copy_options::overwrite_existing);
        };
        cp_ready("trial_metric_distributions.csv", "paired_metrics_long.csv");
        cp_ready("paired_delta_long_by_trial.csv", "paired_delta_long.csv");
        cp_ready("paired_delta_by_trial.csv", "paired_delta_by_trial.csv");
        cp_ready("tau_mse_scatter.csv", "tau_mse_scatter.csv");
        cp_ready("temperature_metric_long.csv", "temperature_metrics_long.csv");
        cp_ready("temperature_metric_summary.csv", "temperature_metric_summary.csv");
        cp_ready("temperature_selection_counts.csv", "temperature_selection_counts.csv");
        cp_ready("ci_interval_summary.csv", "uncertainty_interval_bars.csv");
        cp_ready("paired_delta_interval_summary.csv", "delta_interval_bars.csv");
        std::ofstream manifest(plot_ready / "README_plot_ready.txt");
        manifest << "paired_metrics_long.csv: MC-selected and random tau/MSE/score by UQ trial.\n";
        manifest << "paired_delta_long.csv and paired_delta_by_trial.csv: paired deltas; positive values favor CEMC.\n";
        manifest << "tau_mse_scatter.csv: tau-MSE joint scatter; higher tau and lower MSE are better.\n";
        manifest << "temperature_metrics_long.csv: CEMC tau/MSE/score across snapshot temperatures.\n";
        manifest << "temperature_metric_summary.csv: mean/CI/quantile summaries by CEMC temperature.\n";
        manifest << "temperature_selection_counts.csv: selected CEMC temperature counts.\n";
        manifest << "uncertainty_interval_bars.csv and delta_interval_bars.csv: 95% CI interval data.\n";
    }
}



static double sample_sd_from_sums(double sum, double sumsq, int count) {
    if (count <= 1) return 0.0;
    const double mean = sum / static_cast<double>(count);
    double var = (sumsq - static_cast<double>(count) * mean * mean) / static_cast<double>(count - 1);
    if (var < 0.0 && var > -1e-10) var = 0.0;
    if (var < 0.0) var = 0.0;
    return std::sqrt(var);
}

static void merge_trial(
    const fs::path &outdir, const fs::path &trial_dir, int trial, int world_size,
    const SchedulePlan &plan, const std::vector<CompositionRow> &rows,
    const Config &cfg, int n_metal
) {
    fs::path marker = outdir / "merge_in_progress_trial.txt";
    { std::ofstream m(marker); m << trial << "\n"; }

    const int ntemps = static_cast<int>(plan.target_temperatures.size());
    const int ncomp = static_cast<int>(rows.size());
    auto cidx = [&](int ti, int ci) { return ti * ncomp + ci; };
    std::vector<double> cemc_sums(static_cast<size_t>(ntemps)*ncomp, 0.0);
    std::vector<double> cemc_sumsq(static_cast<size_t>(ntemps)*ncomp, 0.0);
    std::vector<int> cemc_counts(static_cast<size_t>(ntemps)*ncomp, 0);
    std::vector<double> random_sums(ncomp, 0.0);
    std::vector<double> random_sumsq(ncomp, 0.0);
    std::vector<int> random_counts(ncomp, 0);

    for (int rank=0; rank<world_size; ++rank) {
        std::ostringstream rp;
        rp << "rank_" << std::setw(5) << std::setfill('0') << rank << "_results.csv";
        fs::path p = trial_dir / rp.str();
        if (!fs::exists(p)) continue;
        std::ifstream in(p);
        std::string line;
        while (std::getline(in, line)) {
            if (trim(line).empty()) continue;
            auto r = parse_result_line(line);
            if (r.comp < 0 || r.comp >= ncomp) continue;
            if (r.method == "cemc") {
                if (r.temp_idx < 0 || r.temp_idx >= ntemps) continue;
                cemc_sums[cidx(r.temp_idx, r.comp)] += r.activity;
                cemc_sumsq[cidx(r.temp_idx, r.comp)] += r.activity * r.activity;
                cemc_counts[cidx(r.temp_idx, r.comp)] += 1;
            } else if (r.method == "random") {
                random_sums[r.comp] += r.activity;
                random_sumsq[r.comp] += r.activity * r.activity;
                random_counts[r.comp] += 1;
            }
        }
    }

    std::vector<double> exp_raw(rows.size());
    for (size_t i=0;i<rows.size();i++) exp_raw[i] = rows[i].experimental;
    auto exp_scaled = scale_to_minus1_0(exp_raw, !cfg.experimental_better_is_lower);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> cemc_mean(static_cast<size_t>(ntemps)*ncomp, nan);
    std::vector<double> cemc_sd(static_cast<size_t>(ntemps)*ncomp, nan);
    std::vector<double> cemc_scaled(static_cast<size_t>(ntemps)*ncomp, nan);
    std::vector<double> random_mean(ncomp, nan);
    std::vector<double> random_sd(ncomp, nan);
    std::vector<double> random_scaled_vec(ncomp, nan);

    std::unique_ptr<std::ofstream> paselected;
    if (cfg.write_predicted_activities) {
        paselected = std::make_unique<std::ofstream>(outdir / "predicted_activity_selected_by_trial.csv", std::ios::app);
        (*paselected) << std::setprecision(12);
    }

    std::ofstream metrics(outdir / "metrics_by_trial_temperature.csv", std::ios::app);
    metrics << std::setprecision(12);
    std::vector<Metric> cemc_metrics(ntemps);
    for (int ti=0; ti<ntemps; ++ti) {
        std::vector<double> pred(rows.size(), 0.0);
        std::vector<double> pred_sd(rows.size(), 0.0);
        int record_count = 0;
        for (int ci=0; ci<ncomp; ++ci) {
            int c = cemc_counts[cidx(ti, ci)];
            if (c > 0) {
                pred[ci] = cemc_sums[cidx(ti, ci)] / c;
                pred_sd[ci] = sample_sd_from_sums(cemc_sums[cidx(ti, ci)], cemc_sumsq[cidx(ti, ci)], c);
            }
            record_count += c;
        }
        auto pred_scaled = scale_to_minus1_0(pred, cfg.predicted_better_is_higher);
        double tau = kendall_tau_b(exp_scaled, pred_scaled);
        double mmse = mse(exp_scaled, pred_scaled);
        cemc_metrics[ti] = {tau, mmse};
        metrics << trial << ',' << ti << ',' << plan.target_temperatures[ti] << ",cemc,"
                << tau << ',' << mmse << ',' << metric_score(cemc_metrics[ti]) << ',' << ncomp << ',' << record_count << "\n";
        for (int ci=0; ci<ncomp; ++ci) {
            cemc_mean[cidx(ti, ci)] = pred[ci];
            cemc_sd[cidx(ti, ci)] = pred_sd[ci];
            cemc_scaled[cidx(ti, ci)] = pred_scaled[ci];

        }
    }
    metrics.close();

    std::vector<double> random_pred(rows.size(), 0.0);
    int random_record_count = 0;
    for (int ci=0; ci<ncomp; ++ci) {
        int c = random_counts[ci];
        if (c > 0) {
            random_pred[ci] = random_sums[ci] / c;
            random_sd[ci] = sample_sd_from_sums(random_sums[ci], random_sumsq[ci], c);
        }
        random_mean[ci] = random_pred[ci];
        random_record_count += c;
    }
    auto random_scaled = scale_to_minus1_0(random_pred, cfg.predicted_better_is_higher);
    for (int ci=0; ci<ncomp; ++ci) {
        random_scaled_vec[ci] = random_scaled[ci];

    }
    Metric random_metric{kendall_tau_b(exp_scaled, random_scaled), mse(exp_scaled, random_scaled)};
    {
        std::ofstream rm(outdir / "random_metrics_by_trial.csv", std::ios::app);
        rm << std::setprecision(12)
           << trial << ",random," << random_metric.tau << ',' << random_metric.mse << ','
           << metric_score(random_metric) << ',' << ncomp << ',' << random_record_count << "\n";
    }

    auto best_score_idx = [](const std::vector<Metric> &m) {
        int best = 0;
        for (int i=1;i<(int)m.size();i++) {
            double si = metric_score(m[i]);
            double sb = metric_score(m[best]);
            if (si > sb || (std::abs(si - sb) < 1e-15 && (m[i].tau > m[best].tau || (std::abs(m[i].tau - m[best].tau) < 1e-15 && m[i].mse < m[best].mse)))) best = i;
        }
        return best;
    };
    int cbest = best_score_idx(cemc_metrics);
    Metric selected = cemc_metrics[cbest];
    double selected_score = metric_score(selected);
    double random_score = metric_score(random_metric);
    double delta_tau = selected.tau - random_metric.tau;
    double delta_mse = random_metric.mse - selected.mse;
    double delta_score = selected_score - random_score;
    {
        std::ofstream sel(outdir / "selected_temperature_by_trial.csv", std::ios::app);
        sel << std::setprecision(12)
            << trial << ','
            << plan.target_temperatures[cbest] << ',' << selected_score << ',' << selected.tau << ',' << selected.mse << ','
            << random_score << ',' << random_metric.tau << ',' << random_metric.mse << ','
            << delta_tau << ',' << delta_mse << ',' << delta_score << "\n";
    }
    {
        std::ofstream paired(outdir / "paired_comparison_by_trial.csv", std::ios::app);
        paired << std::setprecision(12)
            << trial << ',' << plan.target_temperatures[cbest] << ','
            << selected.tau << ',' << selected.mse << ',' << selected_score << ','
            << random_metric.tau << ',' << random_metric.mse << ',' << random_score << ','
            << delta_tau << ',' << delta_mse << ',' << delta_score << "\n";
    }

    if (cfg.write_predicted_activities) {
        for (int ci=0; ci<ncomp; ++ci) {
            (*paselected) << trial << ',' << ci << ',' << rows[ci].row_idx << ',' << rows[ci].col_idx << ','
                          << cbest << ',' << plan.target_temperatures[cbest] << ','
                          << cemc_mean[cidx(cbest, ci)] << ',' << cemc_sd[cidx(cbest, ci)] << ',' << cemc_counts[cidx(cbest, ci)] << ','
                          << random_mean[ci] << ',' << random_sd[ci] << ',' << random_counts[ci] << ','
                          << rows[ci].experimental << ',' << cemc_scaled[cidx(cbest, ci)] << ','
                          << random_scaled_vec[ci] << ',' << exp_scaled[ci] << "\n";
        }
    }

    append_trial_samples(outdir, trial, rows, cfg, n_metal);
    if (cfg.write_random_seeds) append_random_seed_shards(outdir, trial_dir, world_size);

    if (cfg.write_structures) {
        std::set<fs::path> all_possible_paths;
        for (int ti=0; ti<ntemps; ++ti) for (int ci=0; ci<ncomp; ++ci)
            all_possible_paths.insert(structure_file_path(outdir, plan.target_temperatures[ti], ci));
        {
            std::ofstream marker2(outdir / "merge_in_progress_trial.txt");
            marker2 << trial << "\n";
            for (const auto &p : all_possible_paths) marker2 << p.string() << "\n";
        }
        struct FileCacheEntry { fs::path path; std::unique_ptr<std::ofstream> out; uint64_t last_use = 0; };
        std::vector<FileCacheEntry> cache;
        const size_t max_open_files = 64;
        uint64_t use_counter = 0;
        auto get_out = [&](const fs::path &path) -> std::ofstream& {
            ++use_counter;
            for (auto &e : cache) if (e.path == path) { e.last_use = use_counter; return *e.out; }
            if (cache.size() >= max_open_files) {
                auto victim = std::min_element(cache.begin(), cache.end(), [](const auto &a, const auto &b){ return a.last_use < b.last_use; });
                victim->out->close();
                cache.erase(victim);
            }
            fs::create_directories(path.parent_path());
            FileCacheEntry e;
            e.path = path;
            e.out = std::make_unique<std::ofstream>(path, std::ios::app);
            e.last_use = use_counter;
            cache.push_back(std::move(e));
            return *cache.back().out;
        };
        for (int rank=0; rank<world_size; ++rank) {
            std::ostringstream rp;
            rp << "rank_" << std::setw(5) << std::setfill('0') << rank << "_structures.jsonl";
            fs::path p = trial_dir / rp.str();
            if (!fs::exists(p)) continue;
            std::ifstream in(p);
            std::string line;
            while (std::getline(in, line)) {
                auto tab1 = line.find('\t');
                if (tab1 == std::string::npos) continue;
                auto tab2 = line.find('\t', tab1 + 1);
                if (tab2 == std::string::npos) continue;
                int ti = std::stoi(line.substr(0, tab1));
                int ci = std::stoi(line.substr(tab1 + 1, tab2 - tab1 - 1));
                if (ti < 0 || ti >= ntemps || ci < 0 || ci >= ncomp) continue;
                auto &out = get_out(structure_file_path(outdir, plan.target_temperatures[ti], ci));
                out << line.substr(tab2 + 1) << "\n";
            }
        }
        for (auto &e : cache) if (e.out) e.out->close();
    }

    {
        std::ofstream done(outdir / "completed_trials.txt", std::ios::app);
        done << trial << "\n";
    }
    fs::remove(marker);
    fs::remove_all(trial_dir);
}


static double seconds_since(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

static void append_trial_timing(const fs::path &outdir, int trial, double compute_s, double merge_s, double summary_s,
                                int64_t n_tasks, int world_size, bool structures_written, bool summary_written) {
    std::ofstream out(outdir / "trial_timing.csv", std::ios::app);
    out << std::setprecision(6) << std::fixed
        << trial << ',' << compute_s << ',' << merge_s << ',' << summary_s << ',' << (compute_s + merge_s + summary_s) << ','
        << n_tasks << ',' << world_size << ',' << (structures_written ? "true" : "false") << ',' << (summary_written ? "true" : "false") << "\n";
}

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank=0, world_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    int rank=0, world_size=1;
#endif
    try {
        std::string config_path;
        bool reconstruct_random = false;
        bool reconstruct_cemc = false;
        int recon_trial = -1, recon_comp = -1, recon_run = -1;
        int recon_temp_index = -1;
        double recon_temperature = std::numeric_limits<double>::quiet_NaN();
        bool have_manual_seed = false, have_manual_counts = false;
        uint64_t manual_seed = 0;
        std::array<int,5> manual_counts{};
        std::string recon_output = "random_slab_atomic_numbers.json";
        for (int i=1; i<argc; ++i) {
            std::string a = argv[i];
            if (a == "--config" && i+1 < argc) config_path = argv[++i];
            else if (a == "--reconstruct-random") reconstruct_random = true;
            else if (a == "--reconstruct-cemc") reconstruct_cemc = true;
            else if (a == "--trial" && i+1 < argc) recon_trial = std::stoi(argv[++i]);
            else if (a == "--composition" && i+1 < argc) recon_comp = std::stoi(argv[++i]);
            else if (a == "--run" && i+1 < argc) recon_run = std::stoi(argv[++i]);
            else if (a == "--temp-index" && i+1 < argc) recon_temp_index = std::stoi(argv[++i]);
            else if (a == "--temperature" && i+1 < argc) recon_temperature = std::stod(argv[++i]);
            else if (a == "--seed" && i+1 < argc) { manual_seed = static_cast<uint64_t>(std::stoull(argv[++i])); have_manual_seed = true; }
            else if (a == "--counts" && i+1 < argc) { manual_counts = parse_counts_arg(argv[++i]); have_manual_counts = true; }
            else if (a == "--output" && i+1 < argc) recon_output = argv[++i];
            else if (a == "-h" || a == "--help") {
                if (rank == 0) {
                    std::cout << "usage: uq_cemc_mpi --config uq_config.ini\n"
                              << "       uq_cemc_mpi --config uq_config.ini --reconstruct-random --trial T --composition C --run R --output slab.json\n"
                              << "       uq_cemc_mpi --config uq_config.ini --reconstruct-random --seed S --counts Fe,Co,Ni,Pd,Pt --output slab.json\n"
                              << "       uq_cemc_mpi --config uq_config.ini --reconstruct-cemc --trial T --composition C --run R --temperature 1500 --output snapshot.json\n"
                              << "       uq_cemc_mpi --config uq_config.ini --reconstruct-cemc --trial T --composition C --run R --temp-index I --output snapshot.json\n";
                }
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + a);
            }
        }
        if (config_path.empty()) throw std::runtime_error("Missing --config uq_config.ini");
        Config cfg = read_config(config_path);
        CEData ce = read_ce_export(cfg.ce_export);
        auto plan = read_schedule_export(cfg.schedule_export);
        ActivityModel am = read_activity_model(cfg.activity_model, ce);
        auto rows = read_compositions_and_activity(cfg.composition_csv, cfg.experimental_activity_csv, cfg.max_compositions);
        if (rows.empty()) throw std::runtime_error("No composition/activity rows loaded");
        fs::path outdir(cfg.output_dir);
        if (rank == 0) {
            fs::create_directories(outdir);
            repair_partial_merge(outdir, plan);
            ensure_headers(outdir);
            std::cerr << "Loaded " << rows.size() << " compositions; " << plan.target_temperatures.size() << " CEMC snapshot target temperatures; "
                      << cfg.n_trials << " trials; " << cfg.n_runs << " runs; MPI ranks=" << world_size << "\n";
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        std::set<int> completed = read_completed_trials(outdir);
        const std::array<std::string,5> elems = {"Fe","Co","Ni","Pd","Pt"};
        std::array<int,5> element_to_code{};
        for (int k=0;k<5;k++) {
            auto it = ce.species_to_code.find(elems[k]);
            if (it == ce.species_to_code.end()) throw std::runtime_error("CE species missing: " + elems[k]);
            element_to_code[k] = it->second;
        }

        if (reconstruct_random) {
            if (rank == 0) {
                uint64_t seed = manual_seed;
                std::array<int,5> counts = manual_counts;
                if (!(have_manual_seed && have_manual_counts)) {
                    if (recon_trial < 0 || recon_comp < 0 || recon_run < 0)
                        throw std::runtime_error("For --reconstruct-random, provide either --seed and --counts, or --trial/--composition/--run");
                    auto rec = find_random_seed_record(fs::path(cfg.output_dir), recon_trial, recon_comp, recon_run);
                    seed = rec.random_slab_seed;
                    counts = rec.counts;
                    std::cerr << "Reconstructing random slab from seed row: trial=" << rec.trial
                              << " composition=" << rec.comp << " run=" << rec.run
                              << " seed=" << seed << "\n";
                }
                reconstruct_random_slab_file(ce, element_to_code, seed, counts, fs::path(recon_output));
                std::cerr << "Wrote " << recon_output << "\n";
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        }

        if (reconstruct_cemc) {
            if (rank == 0) {
                if (recon_trial < 0 || recon_comp < 0 || recon_run < 0)
                    throw std::runtime_error("For --reconstruct-cemc, provide --trial T --composition C --run R plus --temperature or --temp-index");
                if (recon_comp < 0 || recon_comp >= static_cast<int>(rows.size()))
                    throw std::runtime_error("--composition is outside the loaded composition range");
                int temp_index = recon_temp_index;
                if (temp_index < 0) {
                    if (!std::isfinite(recon_temperature))
                        throw std::runtime_error("For --reconstruct-cemc, provide --temperature or --temp-index");
                    double best_diff = std::numeric_limits<double>::infinity();
                    for (int i=0; i<static_cast<int>(plan.target_temperatures.size()); ++i) {
                        double d = std::abs(plan.target_temperatures[i] - recon_temperature);
                        if (d < best_diff) { best_diff = d; temp_index = i; }
                    }
                }
                if (temp_index < 0 || temp_index >= static_cast<int>(plan.target_temperatures.size()))
                    throw std::runtime_error("Temperature index is outside the target-temperature list");
                const auto &row = rows[recon_comp];
                auto sampled = sample_composition(row, cfg, ce.n_metal_sites, recon_trial, recon_comp);
                uint64_t initial_seed = random_slab_seed(cfg, recon_trial, recon_comp, recon_run);
                uint64_t mc_seed = cemc_seed(cfg, recon_trial, recon_comp, recon_run);
                std::mt19937_64 init_rng(initial_seed);
                auto initial_occ = make_initial_occupations(ce, sampled, element_to_code, init_rng);
                std::mt19937_64 mc_rng(mc_seed);
                MCOutcome mc = run_cemc_snapshots(ce, initial_occ, plan, mc_rng);
                const Snapshot &snap = mc.snapshots[temp_index];
                double activity = predict_activity(ce, am, snap.occ, sampled.be_shift);
                fs::path outp(recon_output);
                if (!outp.parent_path().empty()) fs::create_directories(outp.parent_path());
                std::ofstream out(outp);
                out << "{\n";
                out << "  \"trial\": " << recon_trial << ",\n";
                out << "  \"composition_index\": " << recon_comp << ",\n";
                out << "  \"row_idx\": " << row.row_idx << ",\n";
                out << "  \"col_idx\": " << row.col_idx << ",\n";
                out << "  \"run\": " << recon_run << ",\n";
                out << "  \"temp_index\": " << temp_index << ",\n";
                out << "  \"target_temperature\": " << std::setprecision(12) << plan.target_temperatures[temp_index] << ",\n";
                out << "  \"actual_temperature\": " << std::setprecision(12) << snap.actual_temperature << ",\n";
                out << "  \"temperature_error\": " << std::setprecision(12) << snap.temperature_error << ",\n";
                out << "  \"mc_step\": " << snap.global_step << ",\n";
                out << "  \"energy\": " << std::setprecision(12) << snap.energy << ",\n";
                out << "  \"attempted\": " << snap.attempted << ",\n";
                out << "  \"accepted\": " << snap.accepted << ",\n";
                out << "  \"random_slab_seed\": " << initial_seed << ",\n";
                out << "  \"cemc_seed\": " << mc_seed << ",\n";
                out << "  \"be_shift_by_element\": {";
                for (int k=0; k<5; ++k) {
                    if (k) out << ", ";
                    out << "\"" << am.elements[k] << "\": " << std::setprecision(12) << sampled.be_shift[k];
                }
                out << "},\n";
                out << "  \"activity\": " << std::setprecision(12) << activity << ",\n";
                out << "  \"Z\": [";
                auto z = physical_atomic_numbers(ce, snap.occ);
                for (size_t i=0; i<z.size(); ++i) { if (i) out << ','; out << z[i]; }
                out << "]\n}\n";
                std::cerr << "Reconstructed CEMC snapshot trial=" << recon_trial
                          << " composition=" << recon_comp << " run=" << recon_run
                          << " target_temperature=" << plan.target_temperatures[temp_index]
                          << " to " << recon_output << "\n";
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        }

        for (int trial=0; trial<cfg.n_trials; ++trial) {
            if (completed.count(trial)) {
                if (rank == 0) std::cerr << "Skipping completed trial " << trial << "\n";
                continue;
            }
            fs::path trial_dir = outdir / "_trial_work" / ("trial_" + std::to_string(trial));
            auto trial_wall_start = std::chrono::steady_clock::now();
            if (rank == 0) {
                fs::remove_all(trial_dir);
                fs::create_directories(trial_dir);
                std::cerr << "Starting trial " << trial << "\n";
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            auto compute_start = std::chrono::steady_clock::now();
            fs::create_directories(trial_dir);
            std::ostringstream rp;
            rp << "rank_" << std::setw(5) << std::setfill('0') << rank;
            std::ofstream results(trial_dir / (rp.str() + "_results.csv"));
            std::ofstream seed_records;
            if (cfg.write_random_seeds) seed_records.open(trial_dir / (rp.str() + "_seeds.csv"));
            std::ofstream structs;
            if (cfg.write_structures) structs.open(trial_dir / (rp.str() + "_structures.jsonl"));
            results << std::setprecision(12);
            if (seed_records.is_open()) seed_records << std::setprecision(12);
            const int ncomp = static_cast<int>(rows.size());
            const int ntemps = static_cast<int>(plan.target_temperatures.size());
            // One MPI task = one independent CEMC trajectory for one
            // (trial, composition, run). The original schedule.yaml is run once;
            // 2000/1900/.../300 K configurations are snapshots from that trajectory.
            int64_t n_tasks = static_cast<int64_t>(ncomp) * cfg.n_runs;
            for (int64_t task = rank; task < n_tasks; task += world_size) {
                int run_idx = static_cast<int>(task % cfg.n_runs);
                int comp_idx = static_cast<int>(task / cfg.n_runs);
                const auto &row = rows[comp_idx];
                auto sampled = sample_composition(row, cfg, ce.n_metal_sites, trial, comp_idx);
                uint64_t comp_seed = composition_seed(cfg, trial, comp_idx);
                uint64_t initial_seed = random_slab_seed(cfg, trial, comp_idx, run_idx);
                uint64_t mc_seed = cemc_seed(cfg, trial, comp_idx, run_idx);
                std::mt19937_64 init_rng(initial_seed);
                auto initial_occ = make_initial_occupations(ce, sampled, element_to_code, init_rng);
                if (seed_records.is_open()) {
                    seed_records << trial << ',' << comp_idx << ',' << row.row_idx << ',' << row.col_idx << ',' << run_idx << ','
                                 << comp_seed << ',' << initial_seed << ',' << mc_seed << ',' << cfg.random_seed
                                 << ",initial_random_slab,mt19937_64_deterministic_fisher_yates_v1," << ce.n_metal_sites;
                    for (int x : sampled.counts) seed_records << ',' << x;
                    for (double x : sampled.fraction) seed_records << ',' << x;
                    for (double x : sampled.be_shift) seed_records << ',' << x;
                    seed_records << "\n";
                }
                double random_activity = predict_activity(ce, am, initial_occ, sampled.be_shift);
                // Random slabs have no CEMC temperature. Record exactly one random baseline
                // activity per (trial, composition, run). The seed/counts in random_slab_seeds.csv
                // are sufficient to reconstruct the random slab later.
                results << trial << ',' << comp_idx << ',' << run_idx << ",-1,-1,random," << random_activity << ",0,0,0,0,0\n";
                std::mt19937_64 mc_rng(mc_seed);
                MCOutcome mc = run_cemc_snapshots(ce, initial_occ, plan, mc_rng);
                for (int temp_idx=0; temp_idx<ntemps; ++temp_idx) {
                    const Snapshot &snap = mc.snapshots[temp_idx];
                    double cemc_activity = predict_activity(ce, am, snap.occ, sampled.be_shift);
                    results << trial << ',' << comp_idx << ',' << run_idx << ',' << temp_idx << ','
                            << plan.target_temperatures[temp_idx] << ",cemc," << cemc_activity << ','
                            << snap.energy << ',' << snap.attempted << ',' << snap.accepted << ','
                            << snap.actual_temperature << ',' << snap.global_step << "\n";
                    if (cfg.write_structures) {
                        write_structure_json(structs, trial, comp_idx, row, run_idx, temp_idx, plan.target_temperatures[temp_idx],
                                             "cemc", cemc_activity, &snap, ce, snap.occ);
                    }
                }
            }
            results.close();
            if (seed_records.is_open()) seed_records.close();
            if (structs.is_open()) structs.close();
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            auto compute_end = std::chrono::steady_clock::now();
            if (rank == 0) {
                auto merge_start = std::chrono::steady_clock::now();
                merge_trial(outdir, trial_dir, trial, world_size, plan, rows, cfg, ce.n_metal_sites);
                auto merge_end = std::chrono::steady_clock::now();
                bool do_summary = (cfg.summary_every_trials > 0) && (((trial + 1) % cfg.summary_every_trials == 0) || (trial + 1 == cfg.n_trials));
                double summary_s = 0.0;
                if (do_summary) {
                    auto summary_start = std::chrono::steady_clock::now();
                    write_uncertainty_summary(outdir, cfg.ml_index);
                    auto summary_end = std::chrono::steady_clock::now();
                    summary_s = seconds_since(summary_start, summary_end);
                }
                double compute_s = seconds_since(compute_start, compute_end);
                double merge_s = seconds_since(merge_start, merge_end);
                int64_t trial_tasks = static_cast<int64_t>(rows.size()) * cfg.n_runs;
                append_trial_timing(outdir, trial, compute_s, merge_s, summary_s, trial_tasks, world_size, cfg.write_structures, do_summary);
                std::cerr << "Finished trial " << trial
                          << " compute_s=" << compute_s
                          << " merge_s=" << merge_s
                          << " summary_s=" << summary_s
                          << " total_s=" << (compute_s + merge_s + summary_s) << "\n";
            }
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
        if (rank == 0) {
            auto summary_start = std::chrono::steady_clock::now();
            write_uncertainty_summary(outdir, cfg.ml_index);
            auto summary_end = std::chrono::steady_clock::now();
            std::cerr << "Final uncertainty/plot summary written in " << seconds_since(summary_start, summary_end) << " s\n";
            std::cerr << "All requested trials are complete. Output: " << outdir << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
#endif
        return 1;
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
