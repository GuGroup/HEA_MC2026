#define main uq_cemc_legacy_main_do_not_call
#include "uq_cemc_mpi.cpp"
#undef main

struct Options {
    std::string config;
    fs::path run_activity_dir;
    int trial_start=0, trial_end=-1;
};

static Options parse_options(int argc,char**argv){
    Options o;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--config"&&i+1<argc)o.config=argv[++i];
        else if(a=="--run-activity-dir"&&i+1<argc)o.run_activity_dir=argv[++i];
        else if(a=="--trial-start"&&i+1<argc)o.trial_start=std::stoi(argv[++i]);
        else if(a=="--trial-end"&&i+1<argc)o.trial_end=std::stoi(argv[++i]);
        else throw std::runtime_error("unknown argument "+a);
    }
    if(o.config.empty()||o.run_activity_dir.empty())
        throw std::runtime_error("required: --config --run-activity-dir");
    return o;
}

static std::string shard_stem(int first,int last){
    std::ostringstream s;
    s<<"trials_"<<std::setw(5)<<std::setfill('0')<<first<<"_"<<std::setw(5)<<last;
    return s.str();
}

static void write_metadata(const fs::path&path,const Config&cfg,int ncomp,int first,int ntrial){
    std::ofstream out(path);
    out<<"{\n  \"format\": \"raw_float32_shard_v1\",\n"
       <<"  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
       <<"  \"layout\": [\"trial\", \"composition\", \"run\"],\n"
       <<"  \"first_trial\": "<<first<<",\n  \"n_trials\": "<<ntrial<<",\n"
       <<"  \"shape\": ["<<ntrial<<", "<<ncomp<<", "<<cfg.n_runs<<"],\n"
       <<"  \"activity_definition\": \"per-random-slab activity before run averaging\",\n"
       <<"  \"expected_bytes\": "<<(uint64_t)ntrial*ncomp*cfg.n_runs*sizeof(float)<<"\n}\n";
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank=0,world=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&world);
    try{
        auto o=parse_options(argc,argv);Config cfg=read_config(o.config);CEData ce=read_ce_export(cfg.ce_export);
        ActivityModel am=read_activity_model(cfg.activity_model,ce);
        auto rows=read_compositions_and_activity(cfg.composition_csv,cfg.experimental_activity_csv,cfg.max_compositions);
        int nc=rows.size(),end=o.trial_end<0?cfg.n_trials:std::min(o.trial_end,cfg.n_trials);
        if(am.elements.size()!=5)throw std::runtime_error("activity model must contain five elements");
        std::array<int,5>map{};for(int k=0;k<5;k++){auto it=ce.species_to_code.find(am.elements[k]);if(it==ce.species_to_code.end())throw std::runtime_error("missing species "+am.elements[k]);map[k]=it->second;}
        if(rank==0)fs::create_directories(o.run_activity_dir);MPI_Barrier(MPI_COMM_WORLD);
        MPI_File out_fh{};int open=-1;
        for(int trial=o.trial_start;trial<end;trial++){
            int first=(trial/40)*40,last=std::min(first+39,cfg.n_trials-1),ntrial=last-first+1;
            if(open!=first){
                if(open>=0){MPI_File_sync(out_fh);MPI_File_close(&out_fh);}
                fs::path p=o.run_activity_dir/("random_activity_"+shard_stem(first,last)+".bin");
                std::string ps=p.string();
                if(MPI_File_open(MPI_COMM_WORLD,(char*)ps.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&out_fh)!=MPI_SUCCESS)throw std::runtime_error("random activity shard open failed");
                MPI_File_set_size(out_fh,(MPI_Offset)((uint64_t)ntrial*nc*cfg.n_runs*sizeof(float)));
                if(rank==0)write_metadata(p.string()+".json",cfg,nc,first,ntrial);
                open=first;MPI_Barrier(MPI_COMM_WORLD);
            }
            std::vector<float>local((size_t)nc*cfg.n_runs,0.0f);
            int64_t tasks=(int64_t)nc*cfg.n_runs;
            for(int64_t task=rank;task<tasks;task+=world){
                int run=task%cfg.n_runs,ci=task/cfg.n_runs;
                auto sampled=sample_composition(rows[ci],cfg,ce.n_metal_sites,sample_trial_uncertainty(cfg,trial));
                std::mt19937_64 rng(random_slab_seed(cfg,trial,ci,run));
                auto occ=make_initial_occupations(ce,sampled,map,rng);
                local[(size_t)ci*cfg.n_runs+run]=static_cast<float>(predict_activity(ce,am,occ,sampled.be_shift));
            }
            std::vector<float>values;if(rank==0)values.resize(local.size());
            MPI_Reduce(local.data(),rank?nullptr:values.data(),(int)local.size(),MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
            if(rank==0){
                MPI_Status st;MPI_Offset off=(MPI_Offset)((uint64_t)(trial-first)*nc*cfg.n_runs*sizeof(float));
                if(MPI_File_write_at(out_fh,off,values.data(),(int)values.size(),MPI_FLOAT,&st)!=MPI_SUCCESS)throw std::runtime_error("random activity shard write failed");
                std::cerr<<"completed random trial "<<trial<<'\n';
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if(open>=0){MPI_File_sync(out_fh);MPI_File_close(&out_fh);}
    }catch(const std::exception&e){
        std::cerr<<"rank "<<rank<<" error: "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
    MPI_Finalize();return 0;
}
