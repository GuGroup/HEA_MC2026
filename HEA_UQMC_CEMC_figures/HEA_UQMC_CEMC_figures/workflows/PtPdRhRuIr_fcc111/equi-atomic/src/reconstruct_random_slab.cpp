// reconstruct_random_slab.cpp
// Rebuild a random initial slab atomic-number list from random_slab_seeds.csv.

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static std::string trim(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) e--;
    return s.substr(b, e-b);
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

template <typename T>
static std::vector<T> read_vec(std::istream &in, size_t n) {
    std::vector<T> v(n);
    for (size_t i=0; i<n; ++i) {
        if (!(in >> v[i])) throw std::runtime_error("Unexpected EOF while reading vector");
    }
    return v;
}

static uint64_t uniform_index_u64(std::mt19937_64 &rng, uint64_t bound) {
    if (bound == 0) throw std::runtime_error("uniform_index_u64 called with bound=0");
    const uint64_t threshold = (uint64_t(0) - bound) % bound;
    while (true) {
        uint64_t r = rng();
        if (r >= threshold) return r % bound;
    }
}

static void deterministic_shuffle(std::vector<int> &v, std::mt19937_64 &rng) {
    for (size_t i = v.size(); i > 1; --i) {
        size_t j = static_cast<size_t>(uniform_index_u64(rng, static_cast<uint64_t>(i)));
        std::swap(v[i - 1], v[j]);
    }
}

struct CEHeader {
    int q = 0;
    int n_metal_sites = 0;
    std::vector<std::string> species;
    std::vector<int> atomic_numbers;
    std::vector<int> metal_sites;
    std::unordered_map<std::string, int> species_to_code;
};

static CEHeader read_ce_header(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open CE export: " + path);
    std::string magic;
    in >> magic;
    if (magic != "FCC_CE_EXPORT_V1") throw std::runtime_error("Bad CE export magic: " + magic);
    CEHeader ce;
    std::string key;
    while (in >> key) {
        if (key == "n_species") in >> ce.q;
        else if (key == "species") {
            ce.species = read_vec<std::string>(in, ce.q);
            ce.species_to_code.clear();
            for (int i=0;i<ce.q;i++) ce.species_to_code[ce.species[i]] = i;
        }
        else if (key == "atomic_numbers") ce.atomic_numbers = read_vec<int>(in, ce.q);
        else if (key == "n_metal_sites") in >> ce.n_metal_sites;
        else if (key == "metal_sites") {
            ce.metal_sites = read_vec<int>(in, ce.n_metal_sites);
            break;
        }
    }
    if (ce.q <= 0 || ce.species.empty() || ce.atomic_numbers.empty() || ce.metal_sites.empty()) {
        throw std::runtime_error("CE export did not contain species, atomic_numbers, and metal_sites");
    }
    return ce;
}

struct SeedRow {
    uint64_t random_slab_seed = 0;
    std::array<int,5> counts{};
};

static int col(const std::map<std::string,int> &m, const std::string &name) {
    auto it = m.find(name);
    if (it == m.end()) throw std::runtime_error("random_slab_seeds.csv missing column: " + name);
    return it->second;
}

static SeedRow find_seed_row(const std::string &path, int trial, int comp, int run) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open seed CSV: " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty seed CSV: " + path);
    auto header = split_csv_line(line);
    std::map<std::string,int> cols;
    for (int i=0;i<(int)header.size();++i) cols[header[i]] = i;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        auto v = split_csv_line(line);
        if ((int)v.size() < (int)header.size()) continue;
        int t = std::stoi(v[col(cols, "trial")]);
        int c = std::stoi(v[col(cols, "composition_index")]);
        int r = std::stoi(v[col(cols, "run")]);
        if (t == trial && c == comp && r == run) {
            SeedRow sr;
            sr.random_slab_seed = static_cast<uint64_t>(std::stoull(v[col(cols, "random_slab_seed")]));
            const std::array<std::string,5> names = {"count_Ir","count_Pd","count_Pt","count_Rh","count_Ru"};
            for (int k=0;k<5;k++) sr.counts[k] = std::stoi(v[col(cols, names[k])]);
            return sr;
        }
    }
    throw std::runtime_error("No matching seed row found for requested trial/composition/run");
}

static void usage() {
    std::cerr << "usage: reconstruct_random_slab --ce-export ce_export.txt --seeds random_slab_seeds.csv "
              << "--trial N --composition-index N --run N [--output random_slab.json]\n";
}

int main(int argc, char **argv) {
    try {
        std::string ce_export, seeds, output;
        int trial = -1, comp = -1, run = -1;
        for (int i=1;i<argc;i++) {
            std::string a = argv[i];
            if (a == "--ce-export" && i+1 < argc) ce_export = argv[++i];
            else if (a == "--seeds" && i+1 < argc) seeds = argv[++i];
            else if (a == "--trial" && i+1 < argc) trial = std::stoi(argv[++i]);
            else if (a == "--composition-index" && i+1 < argc) comp = std::stoi(argv[++i]);
            else if (a == "--run" && i+1 < argc) run = std::stoi(argv[++i]);
            else if (a == "--output" && i+1 < argc) output = argv[++i];
            else if (a == "-h" || a == "--help") { usage(); return 0; }
            else throw std::runtime_error("Unknown or incomplete argument: " + a);
        }
        if (ce_export.empty() || seeds.empty() || trial < 0 || comp < 0 || run < 0) {
            usage();
            return 2;
        }
        CEHeader ce = read_ce_header(ce_export);
        SeedRow sr = find_seed_row(seeds, trial, comp, run);
        const std::array<std::string,5> elems = {"Ir","Pd","Pt","Rh","Ru"};
        std::vector<int> metal_codes;
        metal_codes.reserve(ce.n_metal_sites);
        for (int k=0;k<5;k++) {
            auto it = ce.species_to_code.find(elems[k]);
            if (it == ce.species_to_code.end()) throw std::runtime_error("CE species missing: " + elems[k]);
            for (int c=0; c<sr.counts[k]; ++c) metal_codes.push_back(it->second);
        }
        if ((int)metal_codes.size() != ce.n_metal_sites) {
            std::ostringstream msg;
            msg << "Counts sum to " << metal_codes.size() << ", but CE n_metal_sites is " << ce.n_metal_sites;
            throw std::runtime_error(msg.str());
        }
        std::mt19937_64 rng(sr.random_slab_seed);
        deterministic_shuffle(metal_codes, rng);

        std::ostream *outp = &std::cout;
        std::ofstream fout;
        if (!output.empty()) {
            fs::path parent = fs::path(output).parent_path();
            if (!parent.empty()) fs::create_directories(parent);
            fout.open(output);
            if (!fout) throw std::runtime_error("Could not open output: " + output);
            outp = &fout;
        }
        std::ostream &out = *outp;
        out << "[";
        for (size_t i=0;i<metal_codes.size();++i) {
            if (i) out << ',';
            out << ce.atomic_numbers[metal_codes[i]];
        }
        out << "]\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
