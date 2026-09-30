#define main uq_cemc_original_main_do_not_call
#include "uq_cemc_mpi.cpp"
#undef main

#include <omp.h>

// Recreate deterministic random slabs and store mean(ln(activity)) over runs.
int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: recover_random_logmean CONFIG OUTPUT_DIR THREADS\n";
            return 2;
        }
        const std::string config_path = argv[1];
        const fs::path output_dir = argv[2];
        const int threads = std::max(1, std::stoi(argv[3]));

        Config cfg = read_config(config_path);
        CEData ce = read_ce_export(cfg.ce_export);
        ActivityModel am = read_activity_model(cfg.activity_model, ce);
        auto rows = read_compositions_and_activity(
            cfg.composition_csv, cfg.experimental_activity_csv, cfg.max_compositions
        );
        const std::array<std::string,5> elems = {"Ir", "Pd", "Pt", "Rh", "Ru"};
        std::array<int,5> element_to_code{};
        for (int k=0; k<5; ++k) {
            auto it = ce.species_to_code.find(elems[k]);
            if (it == ce.species_to_code.end()) {
                throw std::runtime_error("CE species missing: " + elems[k]);
            }
            element_to_code[k] = it->second;
        }

        fs::create_directories(output_dir);
        const int shard_trials = 40;
        const int nshards = (cfg.n_trials + shard_trials - 1) / shard_trials;
        omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int shard=0; shard<nshards; ++shard) {
            const int begin = shard * shard_trials;
            const int end = std::min(cfg.n_trials, begin + shard_trials);
            std::ostringstream name;
            name << "trials_" << std::setw(5) << std::setfill('0') << begin
                 << "_" << std::setw(5) << std::setfill('0') << (end - 1) << ".csv";
            const fs::path final_path = output_dir / name.str();
            if (fs::exists(final_path)) continue;
            const fs::path temp_path = final_path.string() + ".partial";
            std::ofstream out(temp_path);
            if (!out) throw std::runtime_error("Could not write " + temp_path.string());
            out << std::setprecision(12);
            out << "trial,composition_index,row_idx,col_idx,random_mean_ln_activity,"
                   "random_n_runs\n";

            for (int trial=begin; trial<end; ++trial) {
                const TrialUncertainty uncertainty = sample_trial_uncertainty(cfg, trial);
                for (int comp_idx=0; comp_idx<static_cast<int>(rows.size()); ++comp_idx) {
                    const auto &row = rows[comp_idx];
                    const auto sampled = sample_composition(
                        row, cfg, ce.n_metal_sites, uncertainty
                    );
                    double sum_log = 0.0;
                    for (int run=0; run<cfg.n_runs; ++run) {
                        std::mt19937_64 rng(random_slab_seed(cfg, trial, comp_idx, run));
                        const auto occ = make_initial_occupations(
                            ce, sampled, element_to_code, rng
                        );
                        const double activity = predict_activity(ce, am, occ, sampled.be_shift);
                        if (!(activity > 0.0) || !std::isfinite(activity)) {
                            throw std::runtime_error("Non-positive/non-finite random activity");
                        }
                        sum_log += std::log(activity);
                    }
                    out << trial << ',' << comp_idx << ',' << row.row_idx << ','
                        << row.col_idx << ',' << (sum_log / cfg.n_runs) << ','
                        << cfg.n_runs << '\n';
                }
            }
            out.close();
            fs::rename(temp_path, final_path);
#pragma omp critical
            std::cerr << "completed " << name.str() << " (" << (shard + 1)
                      << "/" << nshards << ")\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
