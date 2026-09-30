#define main uq_cemc_legacy_main_do_not_call
#include "uq_cemc_mpi.cpp"
#undef main

#include <cstring>

namespace slab3bit {

constexpr std::size_t HEADER_BYTES = 4096;
constexpr char MAGIC[] = "C3SLAB01";

struct Options {
    std::string config_path;
    fs::path output_dir;
    fs::path activity_output_dir;
    int shard_trials = 40;
    int trial_start = 0;
    int trial_end = -1;
    int max_compositions_override = -1;
    int n_runs_override = -1;
    bool write_activity = false;
};

static std::string json_escape(const std::string &s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 0x20) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(c) << std::dec;
        } else out << c;
    }
    return out.str();
}

static std::array<std::string,5> storage_elements(const CEData &ce) {
    if (ce.species_to_code.count("Ir")) return {"Ir","Pd","Pt","Rh","Ru"};
    if (ce.species_to_code.count("Fe")) return {"Fe","Co","Ni","Pd","Pt"};
    throw std::runtime_error("Unsupported five-element system: expected Ir or Fe");
}

static std::array<int,5> element_codes(
    const CEData &ce, const std::array<std::string,5> &elements
) {
    std::array<int,5> out{};
    for (int k=0; k<5; ++k) {
        auto it = ce.species_to_code.find(elements[k]);
        if (it == ce.species_to_code.end()) {
            throw std::runtime_error("CE species missing: " + elements[k]);
        }
        out[k] = it->second;
    }
    return out;
}

static std::vector<unsigned char> pack_occupations_3bit(
    const CEData &ce,
    const std::vector<int> &occ,
    const std::array<int,5> &codes
) {
    const std::size_t nbits = static_cast<std::size_t>(ce.n_metal_sites) * 3u;
    std::vector<unsigned char> packed((nbits + 7u) / 8u, 0);
    std::vector<int> inverse(static_cast<std::size_t>(ce.q), -1);
    for (int k=0; k<5; ++k) {
        if (codes[k] < 0 || codes[k] >= ce.q) {
            throw std::runtime_error("Invalid CE code while packing");
        }
        inverse[static_cast<std::size_t>(codes[k])] = k;
    }
    for (int i=0; i<ce.n_metal_sites; ++i) {
        const int ce_code = occ[ce.metal_sites[static_cast<std::size_t>(i)]];
        if (ce_code < 0 || ce_code >= ce.q ||
            inverse[static_cast<std::size_t>(ce_code)] < 0) {
            throw std::runtime_error("Non-metal or unknown species on a metal site");
        }
        const unsigned value =
            static_cast<unsigned>(inverse[static_cast<std::size_t>(ce_code)]);
        const std::size_t bit = static_cast<std::size_t>(i) * 3u;
        const std::size_t byte = bit >> 3u;
        const unsigned shift = static_cast<unsigned>(bit & 7u);
        packed[byte] |= static_cast<unsigned char>(value << shift);
        if (shift > 5u) {
            packed[byte + 1u] |=
                static_cast<unsigned char>(value >> (8u - shift));
        }
    }
    return packed;
}

static std::string shard_stem(int first_trial, int last_trial) {
    std::ostringstream out;
    out << "trials_" << std::setw(5) << std::setfill('0') << first_trial
        << "_" << std::setw(5) << std::setfill('0') << last_trial;
    return out.str();
}

static std::string header_json(
    const Config &cfg,
    const CEData &ce,
    const SchedulePlan &plan,
    const std::array<std::string,5> &elements,
    int first_trial,
    int n_trials_in_shard,
    int ncomp,
    int bytes_per_slab,
    bool write_activity
) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{"
        << "\"format\":\"CEMC_SLAB_3BIT_V1\","
        << "\"version\":1,"
        << "\"header_bytes\":" << HEADER_BYTES << ","
        << "\"layout\":[\"trial\",\"composition\",\"run\",\"temperature\",\"packed_sites\"],"
        << "\"first_trial\":" << first_trial << ","
        << "\"n_trials_in_shard\":" << n_trials_in_shard << ","
        << "\"n_trials_total\":" << cfg.n_trials << ","
        << "\"n_compositions\":" << ncomp << ","
        << "\"n_runs\":" << cfg.n_runs << ","
        << "\"n_temperatures\":" << plan.target_temperatures.size() << ","
        << "\"n_metal_sites\":" << ce.n_metal_sites << ","
        << "\"bits_per_site\":3,"
        << "\"bytes_per_slab\":" << bytes_per_slab << ","
        << "\"random_seed\":" << cfg.random_seed << ","
        << "\"write_activity\":" << (write_activity ? "true" : "false") << ","
        << "\"activity_value\":\""
        << (write_activity ? "raw_non_log_slab_activity" : "not_written") << "\","
        << "\"elements\":[";
    for (int k=0; k<5; ++k) {
        if (k) out << ',';
        out << "\"" << json_escape(elements[k]) << "\"";
    }
    out << "],\"atomic_numbers\":[";
    for (int k=0; k<5; ++k) {
        if (k) out << ',';
        const int code = ce.species_to_code.at(elements[k]);
        out << ce.atomic_numbers[static_cast<std::size_t>(code)];
    }
    out << "],\"temperatures_K\":[";
    for (std::size_t i=0; i<plan.target_temperatures.size(); ++i) {
        if (i) out << ',';
        out << plan.target_temperatures[i];
    }
    out << "]}";
    return out.str();
}

static void write_header_file(
    const fs::path &path,
    const std::string &json,
    std::uint64_t payload_bytes
) {
    if (json.size() + 16u > HEADER_BYTES) {
        throw std::runtime_error("3-bit shard JSON header exceeds 4096 bytes");
    }
    std::vector<unsigned char> header(HEADER_BYTES, 0);
    std::memcpy(header.data(), MAGIC, 8);
    const std::uint32_t version = 1;
    const std::uint32_t json_size = static_cast<std::uint32_t>(json.size());
    std::memcpy(header.data() + 8, &version, sizeof(version));
    std::memcpy(header.data() + 12, &json_size, sizeof(json_size));
    std::memcpy(header.data() + 16, json.data(), json.size());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create shard: " + path.string());
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    if (!out) throw std::runtime_error("Could not write shard header: " + path.string());
    out.close();
    fs::resize_file(path, HEADER_BYTES + payload_bytes);
}

static std::string activity_header_json(
    const Config &cfg,
    const SchedulePlan &plan,
    int first_trial,
    int n_trials_in_shard,
    int ncomp
) {
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\n"
        << "  \"format\": \"CEMC_SLAB_ACTIVITY_F32_V1\",\n"
        << "  \"dtype\": \"little-endian float32\",\n"
        << "  \"header_bytes\": " << HEADER_BYTES << ",\n"
        << "  \"ml_index\": " << cfg.ml_index << ",\n"
        << "  \"first_trial\": " << first_trial << ",\n"
        << "  \"n_trials_in_shard\": " << n_trials_in_shard << ",\n"
        << "  \"n_compositions\": " << ncomp << ",\n"
        << "  \"n_runs\": " << cfg.n_runs << ",\n"
        << "  \"n_temperatures\": " << plan.target_temperatures.size() << ",\n"
        << "  \"shape\": [" << n_trials_in_shard << ", " << ncomp << ", "
        << cfg.n_runs << ", " << plan.target_temperatures.size() << "],\n"
        << "  \"layout\": [\"trial\", \"composition\", \"run\", \"temperature\"],\n"
        << "  \"data_offset_bytes\": " << HEADER_BYTES << ",\n"
        << "  \"activity\": \"raw non-log individual-slab activity\",\n"
        << "  \"be_shift_mean_eV\": " << cfg.be_error_mean << ",\n"
        << "  \"be_shift_sigma_eV\": " << cfg.be_error_sigma << ",\n"
        << "  \"be_shift_application\": \"E_OH_used = E_OH_linear - shift[top_element]\",\n"
        << "  \"temperatures_K\": [";
    for (std::size_t i=0; i<plan.target_temperatures.size(); ++i) {
        if (i) out << ", ";
        out << plan.target_temperatures[i];
    }
    out << "]\n}\n";
    return out.str();
}

static void write_activity_header_file(
    const fs::path &path,
    const std::string &json,
    std::uint64_t payload_bytes
) {
    if (json.size() >= HEADER_BYTES) {
        throw std::runtime_error("Activity shard JSON header exceeds 4096 bytes");
    }
    std::vector<char> header(HEADER_BYTES, 0);
    std::memcpy(header.data(), json.data(), json.size());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create activity shard: " + path.string());
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.close();
    fs::resize_file(path, HEADER_BYTES + payload_bytes);
}

static void write_root_manifest(
    const fs::path &path,
    const Config &cfg,
    const CEData &ce,
    const SchedulePlan &plan,
    const std::array<std::string,5> &elements,
    int ncomp,
    int shard_trials,
    bool write_activity
) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n"
        << "  \"format\": \"CEMC_SLAB_3BIT_V1\",\n"
        << "  \"layout\": [\"trial\", \"composition\", \"run\", \"temperature\", \"packed_sites\"],\n"
        << "  \"header_bytes\": " << HEADER_BYTES << ",\n"
        << "  \"bits_per_site\": 3,\n"
        << "  \"bytes_per_slab\": " << ((ce.n_metal_sites * 3 + 7) / 8) << ",\n"
        << "  \"n_metal_sites\": " << ce.n_metal_sites << ",\n"
        << "  \"n_trials\": " << cfg.n_trials << ",\n"
        << "  \"trials_per_shard\": " << shard_trials << ",\n"
        << "  \"n_compositions\": " << ncomp << ",\n"
        << "  \"n_runs\": " << cfg.n_runs << ",\n"
        << "  \"write_activity\": " << (write_activity ? "true" : "false") << ",\n"
        << "  \"activity_value\": \""
        << (write_activity ? "raw_non_log_individual_slab_activity_float32" : "not_written") << "\",\n"
        << "  \"elements\": [";
    for (int k=0; k<5; ++k) {
        if (k) out << ", ";
        out << "\"" << json_escape(elements[k]) << "\"";
    }
    out << "],\n  \"atomic_numbers\": [";
    for (int k=0; k<5; ++k) {
        if (k) out << ", ";
        const int code = ce.species_to_code.at(elements[k]);
        out << ce.atomic_numbers[static_cast<std::size_t>(code)];
    }
    out << "],\n  \"temperatures_K\": [";
    for (std::size_t i=0; i<plan.target_temperatures.size(); ++i) {
        if (i) out << ", ";
        out << plan.target_temperatures[i];
    }
    out << "]\n}\n";
    std::ofstream f(path, std::ios::trunc);
    if (!f) throw std::runtime_error("Could not create manifest: " + path.string());
    f << out.str();
}

static void write_activity_csv(
    const fs::path &path,
    const Config &cfg,
    const std::vector<CompositionRow> &rows,
    const SchedulePlan &plan,
    int first_trial,
    int n_trials_in_shard,
    const std::vector<double> &mean_sd
) {
    const int ncomp = static_cast<int>(rows.size());
    const int ntemps = static_cast<int>(plan.target_temperatures.size());
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create activity shard: " + path.string());
    out << "trial,composition_index,row_idx,col_idx,temp_idx,target_temperature,"
           "cemc_pred_activity_mean,cemc_pred_activity_sd,cemc_n_runs,"
           "experimental_activity,cemc_pred_activity_scaled,experimental_activity_scaled\n";
    out << std::setprecision(12);
    std::vector<double> exp_raw(static_cast<std::size_t>(ncomp));
    for (int ci=0; ci<ncomp; ++ci) exp_raw[static_cast<std::size_t>(ci)] = rows[ci].experimental;
    const auto exp_scaled =
        scale_to_minus1_0(exp_raw, !cfg.experimental_better_is_lower);
    for (int lt=0; lt<n_trials_in_shard; ++lt) {
        const int trial = first_trial + lt;
        for (int ti=0; ti<ntemps; ++ti) {
            std::vector<double> pred(static_cast<std::size_t>(ncomp));
            for (int ci=0; ci<ncomp; ++ci) {
                const std::size_t idx =
                    (static_cast<std::size_t>(lt) * ncomp + ci) * ntemps + ti;
                pred[static_cast<std::size_t>(ci)] = mean_sd[2u * idx];
            }
            const auto pred_scaled =
                scale_to_minus1_0(pred, cfg.predicted_better_is_higher);
            for (int ci=0; ci<ncomp; ++ci) {
                const std::size_t idx =
                    (static_cast<std::size_t>(lt) * ncomp + ci) * ntemps + ti;
                out << trial << ',' << ci << ',' << rows[ci].row_idx << ','
                    << rows[ci].col_idx << ',' << ti << ','
                    << plan.target_temperatures[static_cast<std::size_t>(ti)] << ','
                    << mean_sd[2u * idx] << ',' << mean_sd[2u * idx + 1u] << ','
                    << cfg.n_runs << ',' << rows[ci].experimental << ','
                    << pred_scaled[static_cast<std::size_t>(ci)] << ','
                    << exp_scaled[static_cast<std::size_t>(ci)] << '\n';
            }
        }
    }
}

static Options parse_options(int argc, char **argv, int rank) {
    Options opt;
    for (int i=1; i<argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i+1 < argc) opt.config_path = argv[++i];
        else if (arg == "--output-dir" && i+1 < argc) opt.output_dir = argv[++i];
        else if (arg == "--activity-output-dir" && i+1 < argc) {
            opt.activity_output_dir = argv[++i];
        } else if (arg == "--shard-trials" && i+1 < argc) {
            opt.shard_trials = std::stoi(argv[++i]);
        } else if (arg == "--trial-start" && i+1 < argc) {
            opt.trial_start = std::stoi(argv[++i]);
        } else if (arg == "--trial-end" && i+1 < argc) {
            opt.trial_end = std::stoi(argv[++i]);
        } else if (arg == "--max-compositions" && i+1 < argc) {
            opt.max_compositions_override = std::stoi(argv[++i]);
        } else if (arg == "--n-runs" && i+1 < argc) {
            opt.n_runs_override = std::stoi(argv[++i]);
        } else if (arg == "--write-activity") opt.write_activity = true;
        else if (arg == "--no-write-activity") opt.write_activity = false;
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout
                    << "usage: recover_cemc_slabs_3bit_mpi --config FILE "
                       "--output-dir DIR [--shard-trials 40] "
                       "[--trial-start 0] [--trial-end N] "
                       "[--max-compositions N] [--n-runs N] "
                       "[--write-activity --activity-output-dir DIR]\n";
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (opt.config_path.empty()) throw std::runtime_error("Missing --config FILE");
    if (opt.output_dir.empty()) throw std::runtime_error("Missing --output-dir DIR");
    if (opt.shard_trials <= 0) throw std::runtime_error("--shard-trials must be positive");
    if (opt.write_activity && opt.activity_output_dir.empty()) {
        opt.activity_output_dir = opt.output_dir / "all_temperature_activity_no_log";
    }
    return opt;
}

static void run(const Options &opt, int rank, int world_size) {
    Config cfg = read_config(opt.config_path);
    if (opt.max_compositions_override > 0) {
        cfg.max_compositions = opt.max_compositions_override;
    }
    if (opt.n_runs_override > 0) cfg.n_runs = opt.n_runs_override;
    CEData ce = read_ce_export(cfg.ce_export);
    SchedulePlan plan = read_schedule_export(cfg.schedule_export);
    const auto rows = read_compositions_and_activity(
        cfg.composition_csv, cfg.experimental_activity_csv, cfg.max_compositions
    );
    if (rows.empty()) throw std::runtime_error("No compositions loaded");
    if (ce.n_metal_sites <= 0) throw std::runtime_error("No metal sites");
    if (ce.n_metal_sites != 1000) {
        throw std::runtime_error(
            "Expected 1000 metal sites for fixed 375-byte packing; found " +
            std::to_string(ce.n_metal_sites)
        );
    }
    const auto elements = storage_elements(ce);
    const auto codes = element_codes(ce, elements);
    ActivityModel am;
    if (opt.write_activity) am = read_activity_model(cfg.activity_model, ce);

    const int ncomp = static_cast<int>(rows.size());
    const int ntemps = static_cast<int>(plan.target_temperatures.size());
    const int bytes_per_slab = (ce.n_metal_sites * 3 + 7) / 8;
    const std::uint64_t bytes_per_group =
        static_cast<std::uint64_t>(cfg.n_runs) *
        static_cast<std::uint64_t>(ntemps) *
        static_cast<std::uint64_t>(bytes_per_slab);
    const int trial_end =
        opt.trial_end < 0 ? cfg.n_trials : std::min(opt.trial_end, cfg.n_trials);
    if (opt.trial_start < 0 || opt.trial_start >= trial_end) {
        throw std::runtime_error("Invalid trial range");
    }

    if (rank == 0) {
        fs::create_directories(opt.output_dir);
        if (opt.write_activity) fs::create_directories(opt.activity_output_dir);
        write_root_manifest(
            opt.output_dir / "format.json", cfg, ce, plan, elements, ncomp,
            opt.shard_trials, opt.write_activity
        );
        std::cerr << "3-bit CEMC recovery: trials " << opt.trial_start << ".."
                  << trial_end - 1 << ", compositions=" << ncomp
                  << ", runs=" << cfg.n_runs << ", temperatures=" << ntemps
                  << ", ranks=" << world_size << "\n";
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for (int first=opt.trial_start; first<trial_end; first += opt.shard_trials) {
        const int last_exclusive = std::min(first + opt.shard_trials, trial_end);
        const int nshard_trials = last_exclusive - first;
        const std::string stem = shard_stem(first, last_exclusive - 1);
        const fs::path final_path = opt.output_dir / (stem + ".bin");
        const fs::path partial_path = opt.output_dir / (stem + ".bin.partial");
        const fs::path activity_final = opt.activity_output_dir / (stem + ".f32");
        const fs::path activity_partial = opt.activity_output_dir / (stem + ".f32.partial");
        const fs::path activity_mean_final = opt.activity_output_dir / (stem + ".mean.csv");
        const fs::path activity_mean_partial = opt.activity_output_dir / (stem + ".mean.csv.partial");

        int skip = 0;
        if (rank == 0) {
            skip = fs::exists(final_path) && (!opt.write_activity ||
                (fs::exists(activity_final) && fs::exists(activity_mean_final)));
        }
#ifdef USE_MPI
        MPI_Bcast(&skip, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        if (skip) {
            if (rank == 0) std::cerr << "Skipping complete shard " << stem << "\n";
            continue;
        }

        const std::uint64_t ngroups =
            static_cast<std::uint64_t>(nshard_trials) *
            static_cast<std::uint64_t>(ncomp);
        const std::uint64_t payload_bytes = ngroups * bytes_per_group;
        if (rank == 0) {
            fs::remove(partial_path);
            if (opt.write_activity) {
                fs::remove(activity_partial);
                fs::remove(activity_mean_partial);
            }
            write_header_file(
                partial_path,
                header_json(
                    cfg, ce, plan, elements, first, nshard_trials, ncomp,
                    bytes_per_slab, opt.write_activity
                ),
                payload_bytes
            );
            if (opt.write_activity) {
                const std::uint64_t activity_count = ngroups *
                    static_cast<std::uint64_t>(cfg.n_runs) *
                    static_cast<std::uint64_t>(ntemps);
                write_activity_header_file(
                    activity_partial,
                    activity_header_json(cfg, plan, first, nshard_trials, ncomp),
                    activity_count * sizeof(float)
                );
            }
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        const std::uint64_t group_begin =
            ngroups * static_cast<std::uint64_t>(rank) /
            static_cast<std::uint64_t>(world_size);
        const std::uint64_t group_end =
            ngroups * static_cast<std::uint64_t>(rank + 1) /
            static_cast<std::uint64_t>(world_size);
        const std::uint64_t local_groups = group_end - group_begin;
        std::vector<unsigned char> packed(
            static_cast<std::size_t>(local_groups * bytes_per_group)
        );
        std::vector<double> local_activity;
        std::vector<float> local_individual_activity;
        if (opt.write_activity) {
            local_activity.assign(
                static_cast<std::size_t>(local_groups) *
                static_cast<std::size_t>(ntemps) * 2u, 0.0
            );
            local_individual_activity.assign(
                static_cast<std::size_t>(local_groups) *
                static_cast<std::size_t>(cfg.n_runs) *
                static_cast<std::size_t>(ntemps), 0.0f
            );
        }

        std::size_t packed_cursor = 0;
        int cached_trial = -1;
        TrialUncertainty uncertainty;
        for (std::uint64_t local_group=0; local_group<local_groups; ++local_group) {
            const std::uint64_t group = group_begin + local_group;
            const int trial_local = static_cast<int>(group / ncomp);
            const int comp_idx = static_cast<int>(group % ncomp);
            const int trial = first + trial_local;
            if (trial != cached_trial) {
                uncertainty = sample_trial_uncertainty(cfg, trial);
                cached_trial = trial;
            }
            const auto sampled = sample_composition(
                rows[static_cast<std::size_t>(comp_idx)], cfg,
                ce.n_metal_sites, uncertainty
            );
            std::vector<double> sum(static_cast<std::size_t>(ntemps), 0.0);
            std::vector<double> sumsq(static_cast<std::size_t>(ntemps), 0.0);
            for (int run_idx=0; run_idx<cfg.n_runs; ++run_idx) {
                std::mt19937_64 init_rng(
                    random_slab_seed(cfg, trial, comp_idx, run_idx)
                );
                auto initial_occ =
                    make_initial_occupations(ce, sampled, codes, init_rng);
                std::mt19937_64 mc_rng(
                    cemc_seed(cfg, trial, comp_idx, run_idx)
                );
                MCOutcome mc =
                    run_cemc_snapshots(ce, initial_occ, plan, mc_rng);
                for (int ti=0; ti<ntemps; ++ti) {
                    const auto bytes = pack_occupations_3bit(
                        ce, mc.snapshots[static_cast<std::size_t>(ti)].occ, codes
                    );
                    std::memcpy(
                        packed.data() + packed_cursor, bytes.data(), bytes.size()
                    );
                    packed_cursor += bytes.size();
                    if (opt.write_activity) {
                        const double value = predict_activity(
                            ce, am,
                            mc.snapshots[static_cast<std::size_t>(ti)].occ,
                            sampled.be_shift
                        );
                        sum[static_cast<std::size_t>(ti)] += value;
                        sumsq[static_cast<std::size_t>(ti)] += value * value;
                        const float fvalue = static_cast<float>(value);
                        if (!std::isfinite(fvalue) || fvalue < 0.0f) {
                            throw std::runtime_error("Individual activity is outside float32 range");
                        }
                        const std::size_t individual_idx =
                            (static_cast<std::size_t>(local_group) * cfg.n_runs + run_idx) *
                            static_cast<std::size_t>(ntemps) + ti;
                        local_individual_activity[individual_idx] = fvalue;
                    }
                }
            }
            if (opt.write_activity) {
                for (int ti=0; ti<ntemps; ++ti) {
                    const double mean =
                        sum[static_cast<std::size_t>(ti)] / cfg.n_runs;
                    double variance = 0.0;
                    if (cfg.n_runs > 1) {
                        variance = (
                            sumsq[static_cast<std::size_t>(ti)] -
                            static_cast<double>(cfg.n_runs) * mean * mean
                        ) / static_cast<double>(cfg.n_runs - 1);
                        if (variance < 0.0 && variance > -1e-24) variance = 0.0;
                    }
                    const std::size_t aidx =
                        (static_cast<std::size_t>(local_group) * ntemps + ti) * 2u;
                    local_activity[aidx] = mean;
                    local_activity[aidx + 1u] =
                        variance > 0.0 ? std::sqrt(variance) : 0.0;
                }
            }
        }
        if (packed_cursor != packed.size()) {
            throw std::runtime_error("Internal packed-buffer size mismatch");
        }

#ifdef USE_MPI
        MPI_File fh;
        const std::string partial_path_string = partial_path.string();
        const int open_rc = MPI_File_open(
            MPI_COMM_WORLD,
            const_cast<char*>(partial_path_string.c_str()),
            MPI_MODE_WRONLY,
            MPI_INFO_NULL,
            &fh
        );
        if (open_rc != MPI_SUCCESS) {
            throw std::runtime_error("MPI_File_open failed for " + partial_path.string());
        }
        const MPI_Offset offset =
            static_cast<MPI_Offset>(HEADER_BYTES) +
            static_cast<MPI_Offset>(group_begin * bytes_per_group);
        if (packed.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            MPI_File_close(&fh);
            throw std::runtime_error("Per-rank packed buffer exceeds MPI int count");
        }
        MPI_Status status;
        const int write_rc = MPI_File_write_at_all(
            fh, offset, packed.data(), static_cast<int>(packed.size()),
            MPI_BYTE, &status
        );
        MPI_File_sync(fh);
        MPI_File_close(&fh);
        if (write_rc != MPI_SUCCESS) {
            throw std::runtime_error("MPI_File_write_at_all failed");
        }
#else
        std::fstream out(partial_path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(static_cast<std::streamoff>(HEADER_BYTES + group_begin * bytes_per_group));
        out.write(reinterpret_cast<const char*>(packed.data()),
                  static_cast<std::streamsize>(packed.size()));
        out.close();
#endif

        if (opt.write_activity) {
#ifdef USE_MPI
            MPI_File afh;
            const std::string activity_partial_string = activity_partial.string();
            const int activity_open_rc = MPI_File_open(
                MPI_COMM_WORLD,
                const_cast<char*>(activity_partial_string.c_str()),
                MPI_MODE_WRONLY,
                MPI_INFO_NULL,
                &afh
            );
            if (activity_open_rc != MPI_SUCCESS) {
                throw std::runtime_error("MPI_File_open failed for " + activity_partial.string());
            }
            const MPI_Offset activity_offset = static_cast<MPI_Offset>(HEADER_BYTES) +
                static_cast<MPI_Offset>(group_begin) * cfg.n_runs * ntemps * sizeof(float);
            if (local_individual_activity.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                MPI_File_close(&afh);
                throw std::runtime_error("Per-rank activity buffer exceeds MPI int count");
            }
            MPI_Status activity_status;
            const int activity_write_rc = MPI_File_write_at_all(
                afh, activity_offset, local_individual_activity.data(),
                static_cast<int>(local_individual_activity.size()), MPI_FLOAT,
                &activity_status
            );
            MPI_File_sync(afh);
            MPI_File_close(&afh);
            if (activity_write_rc != MPI_SUCCESS) {
                throw std::runtime_error("MPI activity write failed");
            }
#else
            std::fstream activity_out(
                activity_partial, std::ios::binary | std::ios::in | std::ios::out
            );
            const std::uint64_t activity_offset = HEADER_BYTES +
                group_begin * static_cast<std::uint64_t>(cfg.n_runs) *
                static_cast<std::uint64_t>(ntemps) * sizeof(float);
            activity_out.seekp(static_cast<std::streamoff>(activity_offset));
            activity_out.write(
                reinterpret_cast<const char*>(local_individual_activity.data()),
                static_cast<std::streamsize>(local_individual_activity.size() * sizeof(float))
            );
            activity_out.close();
#endif
        }

        if (opt.write_activity) {
            const int local_count = static_cast<int>(local_activity.size());
            std::vector<int> counts, displacements;
            std::vector<double> all_activity;
            if (rank == 0) {
                counts.resize(static_cast<std::size_t>(world_size));
                displacements.resize(static_cast<std::size_t>(world_size));
            }
#ifdef USE_MPI
            MPI_Gather(
                &local_count, 1, MPI_INT,
                rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
                0, MPI_COMM_WORLD
            );
            if (rank == 0) {
                int total = 0;
                for (int r=0; r<world_size; ++r) {
                    displacements[static_cast<std::size_t>(r)] = total;
                    total += counts[static_cast<std::size_t>(r)];
                }
                all_activity.resize(static_cast<std::size_t>(total));
            }
            MPI_Gatherv(
                local_activity.data(), local_count, MPI_DOUBLE,
                rank == 0 ? all_activity.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD
            );
#else
            all_activity = local_activity;
#endif
            if (rank == 0) {
                const std::size_t expected =
                    static_cast<std::size_t>(ngroups) *
                    static_cast<std::size_t>(ntemps) * 2u;
                if (all_activity.size() != expected) {
                    throw std::runtime_error("Gathered activity size mismatch");
                }
                write_activity_csv(
                    activity_mean_partial, cfg, rows, plan, first,
                    nshard_trials, all_activity
                );
            }
        }

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (rank == 0) {
            const std::uint64_t expected_size =
                static_cast<std::uint64_t>(HEADER_BYTES) + payload_bytes;
            if (fs::file_size(partial_path) != expected_size) {
                throw std::runtime_error("Packed shard size verification failed");
            }
            fs::remove(final_path);
            fs::rename(partial_path, final_path);
            if (opt.write_activity) {
                const std::uint64_t expected_activity_size =
                    static_cast<std::uint64_t>(HEADER_BYTES) + ngroups *
                    static_cast<std::uint64_t>(cfg.n_runs) *
                    static_cast<std::uint64_t>(ntemps) * sizeof(float);
                if (fs::file_size(activity_partial) != expected_activity_size) {
                    throw std::runtime_error("Individual activity shard size verification failed");
                }
                fs::remove(activity_final);
                fs::rename(activity_partial, activity_final);
                fs::remove(activity_mean_final);
                fs::rename(activity_mean_partial, activity_mean_final);
            }
            std::cerr << "Completed " << final_path
                      << " bytes=" << expected_size << "\n";
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

} // namespace slab3bit

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    int rank = 0, world_size = 1;
#endif
    try {
        const auto opt = slab3bit::parse_options(argc, argv, rank);
        slab3bit::run(opt, rank, world_size);
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
