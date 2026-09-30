#define main uq_cemc_legacy_main_do_not_call
#include "uq_cemc_mpi.cpp"
#undef main

#include <cstring>
#include <regex>

namespace recalc {
constexpr std::size_t HEADER_BYTES = 4096;
constexpr char MAGIC[] = "C3SLAB01";

struct Options {
    std::string config;
    fs::path slab_dir, be_dir, run_activity_dir;
    int trial_start=0, trial_end=-1, trials_per_shard=100;
    double bin_min=-3.0, bin_max=5.0, bin_width=0.01;
};
struct Header { int first=-1,ntrial=-1,ncomp=-1,nrun=-1,ntemp=-1,nmetal=-1,nbyte=-1; };

static int jint(const std::string &s,const std::string &k) {
    std::smatch m; std::regex r("\\\""+k+"\\\"\\s*:\\s*(-?[0-9]+)");
    if(!std::regex_search(s,m,r)) throw std::runtime_error("missing header field "+k);
    return std::stoi(m[1].str());
}
static Header read_header(const fs::path &p) {
    std::ifstream in(p,std::ios::binary); if(!in) throw std::runtime_error("cannot open "+p.string());
    std::array<unsigned char,16> x{}; in.read((char*)x.data(),16);
    if(in.gcount()!=16 || std::memcmp(x.data(),MAGIC,8)!=0) throw std::runtime_error("bad slab header "+p.string());
    uint32_t ver=0,n=0; std::memcpy(&ver,x.data()+8,4); std::memcpy(&n,x.data()+12,4);
    if(ver!=1 || n==0 || n+16>HEADER_BYTES) throw std::runtime_error("unsupported slab header");
    std::string j(n,'\0'); in.read(j.data(),n);
    Header h{jint(j,"first_trial"),jint(j,"n_trials_in_shard"),jint(j,"n_compositions"),
      jint(j,"n_runs"),jint(j,"n_temperatures"),jint(j,"n_metal_sites"),jint(j,"bytes_per_slab")};
    uint64_t expected=HEADER_BYTES+(uint64_t)h.ntrial*h.ncomp*h.nrun*h.ntemp*h.nbyte;
    if(fs::file_size(p)!=expected) throw std::runtime_error("slab shard size mismatch "+p.string());
    return h;
}
static std::string stem(int a,int b) { std::ostringstream s; s<<"trials_"<<std::setw(5)<<std::setfill('0')<<a<<"_"<<std::setw(5)<<b; return s.str(); }
static void unpack(const unsigned char *data,const CEData &ce,const std::array<int,5>&map,std::vector<int>&occ) {
    occ.assign(ce.n_sites,ce.vacancy_code>=0?ce.vacancy_code:0);
    for(int i=0;i<ce.n_metal_sites;i++) { size_t bit=(size_t)i*3, byte=bit>>3; unsigned sh=bit&7;
      unsigned v=(data[byte]>>sh)&7; if(sh>5) v|=(data[byte+1]<<(8-sh))&7;
      if(v>=5) throw std::runtime_error("invalid 3-bit element code"); occ[ce.metal_sites[i]]=map[v]; }
}

static bool site_be(const CEData&ce,const ActivityModel&am,const std::vector<int>&occ,
                    const std::array<double,5>&shift,int s,double&e) {
#ifdef ORR_MODEL
    int site=ce.surface_sites[s], a0=code_to_activity_element(am,ce,occ[site]); if(a0<0) return false;
    e=am.intercept+am.zone1[a0];
    for(int p=am.zone2_ptr[s];p<am.zone2_ptr[s+1];p++){int a=code_to_activity_element(am,ce,occ[am.zone2_indices[p]]);if(a>=0)e+=am.zone2[a];}
    for(int p=am.zone3_ptr[s];p<am.zone3_ptr[s+1];p++){int a=code_to_activity_element(am,ce,occ[am.zone3_indices[p]]);if(a>=0)e+=am.zone3[a];}
    e-=shift[a0]; return true;
#else
    return binding_energy_for_site(ce,am,occ,shift,s,e);
#endif
}
static int nsites(const CEData&ce,const ActivityModel&am) {
#ifdef ORR_MODEL
    return (int)ce.surface_sites.size();
#else
    return (int)am.zone1_ptr.size()-1;
#endif
}

static Options options(int argc,char**argv) { Options o; for(int i=1;i<argc;i++){std::string a=argv[i];
  if(a=="--config"&&i+1<argc)o.config=argv[++i]; else if(a=="--slab-dir"&&i+1<argc)o.slab_dir=argv[++i];
  else if(a=="--be-dir"&&i+1<argc)o.be_dir=argv[++i];
  else if(a=="--run-activity-dir"&&i+1<argc)o.run_activity_dir=argv[++i];
  else if(a=="--trial-start"&&i+1<argc)o.trial_start=std::stoi(argv[++i]); else if(a=="--trial-end"&&i+1<argc)o.trial_end=std::stoi(argv[++i]);
  else if(a=="--trials-per-shard"&&i+1<argc)o.trials_per_shard=std::stoi(argv[++i]);
  else if(a=="--be-bin-min"&&i+1<argc)o.bin_min=std::stod(argv[++i]); else if(a=="--be-bin-max"&&i+1<argc)o.bin_max=std::stod(argv[++i]);
  else if(a=="--be-bin-width"&&i+1<argc)o.bin_width=std::stod(argv[++i]); else throw std::runtime_error("unknown argument "+a); }
  if(o.config.empty()||o.slab_dir.empty()||o.be_dir.empty()||o.run_activity_dir.empty()) throw std::runtime_error("required: --config --slab-dir --be-dir --run-activity-dir");
  return o; }

static void write_shard_metadata(
    const fs::path &path, const Config &cfg,
    const std::vector<CompositionRow> &rows, const SchedulePlan &plan,
    int first_trial, int n_trials
) {
    std::ofstream out(path);
    out << std::setprecision(17)
        << "{\n  \"format\": \"raw_float32_shard_v1\",\n"
        << "  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
        << "  \"layout\": [\"trial\", \"composition\", \"run\", \"temperature\"],\n"
        << "  \"first_trial\": " << first_trial << ",\n"
        << "  \"n_trials\": " << n_trials << ",\n"
        << "  \"shape\": [" << n_trials << ", " << rows.size() << ", "
        << cfg.n_runs << ", " << plan.target_temperatures.size() << "],\n"
        << "  \"temperatures_K\": [";
    for (size_t i=0;i<plan.target_temperatures.size();++i) {
        if (i) out << ", "; out << plan.target_temperatures[i];
    }
    out << "],\n  \"experimental_activity\": [";
    for (size_t i=0;i<rows.size();++i) {
        if (i) out << ", "; out << rows[i].experimental;
    }
    out << "],\n  \"predicted_better_is_higher\": "
        << (cfg.predicted_better_is_higher ? "true" : "false")
        << ",\n  \"experimental_better_is_lower\": "
        << (cfg.experimental_better_is_lower ? "true" : "false")
        << ",\n  \"activity_definition\": \"per-CEMC-slab activity before run averaging\",\n"
        << "  \"expected_bytes\": "
        << (uint64_t)n_trials*rows.size()*cfg.n_runs*plan.target_temperatures.size()*sizeof(float)
        << "\n}\n";
}

static void run(const Options&o,int rank,int world){Config cfg=read_config(o.config);CEData ce=read_ce_export(cfg.ce_export);SchedulePlan plan=read_schedule_export(cfg.schedule_export);
 ActivityModel am=read_activity_model(cfg.activity_model,ce);auto rows=read_compositions_and_activity(cfg.composition_csv,cfg.experimental_activity_csv,cfg.max_compositions);
 const auto &elem=am.elements;if(elem.size()!=5)throw std::runtime_error("activity model must contain five elements");std::array<int,5>map{};for(int k=0;k<5;k++){auto it=ce.species_to_code.find(elem[k]);if(it==ce.species_to_code.end())throw std::runtime_error("missing species "+elem[k]);map[k]=it->second;}
 int nc=rows.size(),nt=plan.target_temperatures.size(),end=o.trial_end<0?cfg.n_trials:std::min(o.trial_end,cfg.n_trials),nb=(int)std::ceil((o.bin_max-o.bin_min)/o.bin_width);
 if(rank==0){fs::create_directories(o.be_dir);fs::create_directories(o.run_activity_dir);std::ofstream m(o.be_dir/"distribution.txt");m<<std::setprecision(12)<<"be_error_mean="<<cfg.be_error_mean<<"\nbe_error_sigma="<<cfg.be_error_sigma<<"\nE_used=E_model-shift\n";}MPI_Barrier(MPI_COMM_WORLD);
 MPI_File slab_fh{},activity_fh{};int open=-1;Header h;std::vector<long long>global_hist((size_t)nt*nb,0);
 for(int trial=o.trial_start;trial<end;trial++){int sf=(trial/40)*40,sl=std::min(sf+39,cfg.n_trials-1);fs::path slab_path=o.slab_dir/(stem(sf,sl)+".bin");
  if(open!=sf){if(open>=0){MPI_File_close(&slab_fh);MPI_File_sync(activity_fh);MPI_File_close(&activity_fh);}if(rank==0)h=read_header(slab_path);MPI_Bcast(&h,sizeof(h),MPI_BYTE,0,MPI_COMM_WORLD);if(h.ncomp!=nc||h.nrun!=cfg.n_runs||h.ntemp!=nt||h.nmetal!=ce.n_metal_sites)throw std::runtime_error("slab/config dimensions mismatch");
   std::string ps=slab_path.string();if(MPI_File_open(MPI_COMM_WORLD,(char*)ps.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&slab_fh)!=MPI_SUCCESS)throw std::runtime_error("slab MPI_File_open failed");fs::path ap=o.run_activity_dir/("activity_"+stem(sf,sl)+".bin");std::string as=ap.string();if(MPI_File_open(MPI_COMM_WORLD,(char*)as.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&activity_fh)!=MPI_SUCCESS)throw std::runtime_error("activity shard open failed");MPI_File_set_size(activity_fh,(MPI_Offset)((uint64_t)h.ntrial*nc*cfg.n_runs*nt*sizeof(float)));if(rank==0)write_shard_metadata(ap.string()+".json",cfg,rows,plan,sf,h.ntrial);open=sf;MPI_Barrier(MPI_COMM_WORLD);}
  size_t bg=(size_t)cfg.n_runs*nt*h.nbyte;std::vector<double>lbe((size_t)nt*3);std::vector<long long>lh((size_t)nt*nb);std::vector<unsigned char>packed(bg);std::vector<int>occ;auto sampled_shift=sample_be_shifts(cfg,trial);
  for(int c=rank;c<nc;c+=world){uint64_t group=(uint64_t)(trial-h.first)*nc+c;MPI_Status st;MPI_File_read_at(slab_fh,HEADER_BYTES+(MPI_Offset)(group*bg),packed.data(),(int)packed.size(),MPI_BYTE,&st);std::vector<float>values((size_t)cfg.n_runs*nt);
   for(int r=0;r<cfg.n_runs;r++)for(int t=0;t<nt;t++){size_t rec=(size_t)r*nt+t;unpack(packed.data()+rec*h.nbyte,ce,map,occ);values[rec]=static_cast<float>(predict_activity(ce,am,occ,sampled_shift));for(int s=0;s<nsites(ce,am);s++){double e;if(!site_be(ce,am,occ,sampled_shift,s,e))continue;size_t q=(size_t)t*3;lbe[q]+=e;lbe[q+1]+=e*e;lbe[q+2]+=1;int b=(int)std::floor((e-o.bin_min)/o.bin_width);if(b>=0&&b<nb)lh[(size_t)t*nb+b]++;}}
   MPI_Offset off=(MPI_Offset)((((uint64_t)(trial-h.first)*nc+c)*cfg.n_runs*nt)*sizeof(float));if(MPI_File_write_at(activity_fh,off,values.data(),(int)values.size(),MPI_FLOAT,&st)!=MPI_SUCCESS)throw std::runtime_error("activity shard write failed");}
  std::vector<double>be;std::vector<long long>hist;if(rank==0){be.resize((size_t)nt*3);hist.resize((size_t)nt*nb);}MPI_Reduce(lbe.data(),rank?nullptr:be.data(),nt*3,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);MPI_Reduce(lh.data(),rank?nullptr:hist.data(),nt*nb,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
  if(rank==0){fs::path bp=o.be_dir/"binding_energy_summary.csv";bool bh=!fs::exists(bp)||fs::file_size(bp)==0;std::ofstream bo(bp,std::ios::app);if(bh)bo<<"trial,temp_idx,temperature,n_binding_energies,be_mean,be_sd\n";bo<<std::setprecision(12);for(int t=0;t<nt;t++){size_t q=(size_t)t*3;double n=be[q+2],mean=be[q]/n,var=n>1?(be[q+1]-be[q]*be[q]/n)/(n-1):0;bo<<trial<<','<<t<<','<<plan.target_temperatures[t]<<','<<(long long)n<<','<<mean<<','<<std::sqrt(std::max(0.0,var))<<'\n';}for(size_t i=0;i<hist.size();i++)global_hist[i]+=hist[i];fs::path sp=o.be_dir/"be_shifts.csv";bool sh=!fs::exists(sp)||fs::file_size(sp)==0;std::ofstream so(sp,std::ios::app);if(sh){so<<"trial";for(auto&e:elem)so<<",be_shift_"<<e;so<<'\n';}so<<trial;for(double x:sampled_shift)so<<','<<std::setprecision(12)<<x;so<<'\n';std::cerr<<"completed trial "<<trial<<'\n';}MPI_Barrier(MPI_COMM_WORLD);
 }
 if(open>=0){MPI_File_close(&slab_fh);MPI_File_sync(activity_fh);MPI_File_close(&activity_fh);}if(rank==0){std::ofstream out(o.be_dir/"be_histogram_by_temperature.csv");out<<"temp_idx,temperature,bin_left,bin_right,count\n";for(int t=0;t<nt;t++)for(int b=0;b<nb;b++)out<<t<<','<<plan.target_temperatures[t]<<','<<o.bin_min+b*o.bin_width<<','<<o.bin_min+(b+1)*o.bin_width<<','<<global_hist[(size_t)t*nb+b]<<'\n';}
}
}
int main(int argc,char**argv){MPI_Init(&argc,&argv);int r=0,w=1;MPI_Comm_rank(MPI_COMM_WORLD,&r);MPI_Comm_size(MPI_COMM_WORLD,&w);try{auto o=recalc::options(argc,argv);recalc::run(o,r,w);}catch(const std::exception&e){std::cerr<<"rank "<<r<<" error: "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}MPI_Finalize();return 0;}
