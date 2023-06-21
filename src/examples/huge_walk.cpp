#include "walk.hpp"
#include "option_helper.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
// #include "dsgl.hpp"
#include "mykernel.cuh"
#include <thread>
#include <stdio.h>
using namespace std;

// extern "C" int cuda_word2vec(int argc, char **argv);
extern "C" int cuda_word2vec (int argc, char **argv, vector<int>* vertex_cn, vector<int>*local_corpus);
struct Empty
{
};

// ./bin/simple_walk -g ./karate.data -v 34 -w 34 -o ./out/walks.txt > perf_dist.txt
int main(int argc, char **argv)
{
    Timer sum_timer;
    // setting of spdlog
    try 
    {
        auto logFactor = spdlog::basic_logger_mt<spdlog::async_factory>("log", "logs/log.txt",true);
        spdlog::set_default_logger(logFactor);
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
    }
    spdlog::set_level(spdlog::level::debug);
    
    Timer timer;

    MPI_Instance mpi_instance(&argc, &argv);

    vector<string> corpus;

    RandomWalkOptionHelper opt;
    opt.parse(argc, argv);

    WalkEngine<real_t, uint32_t> graph;
    //=============== annotation line ===================
    graph.set_init_round(opt.init_round);
    printf("opt min length: %d\n",opt.min_length);
    graph.set_minLength(opt.min_length);
    printf("init_round = %d, min_length = %d\n", graph.init_round, graph.minLength);
    graph.load_graph(opt.v_num, opt.graph_path.c_str(), opt.partition_path.c_str(), opt.make_undirected);
    graph.vertex_cn.resize(graph.get_vertex_num());
    // graph.load_commonNeighbors(opt.graph_common_neighbour.c_str());
    //=============================== launch embed thread ==================

    vector<int> degrees(graph.get_vertex_num());
    for(vertex_id_t i=0; i< graph.v_num;i++)
    {
       degrees[i] = graph.vertex_out_degree[i]; 
    }
    cout<< "============= [trainer_thread launch]================"<<endl;
    thread trainer_thread(cuda_word2vec,argc,argv,&degrees,&graph.local_corpus);
    // cuda_word2vec(argc,argv,&degrees,&graph.local_corpus);

    //======================================================================

    auto extension_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v)
    {
        // return 0.995;
        return walker.step >= 40 ? 0.0 : 1.0;
    };
    auto static_comp = [&](vertex_id_t v, AdjUnit<real_t> *edge)
    {
        return 1.0; /*edge->data is a real number denoting edge weight*/
    };
    auto dynamic_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v, AdjUnit<real_t> *edge)
    {
        return 1.0;
    };
    auto dynamic_comp_upperbound = [&](vertex_id_t v_id, AdjList<real_t> *adj_lists)
    {
        return 1.0;
    };

    WalkerConfig<real_t, uint32_t> walker_conf(opt.walker_num);
    TransitionConfig<real_t, uint32_t> tr_conf(extension_comp);
    for (int i = 0; i < 1; i++) // ???????????? for(int i = 0; i < 1; i++) 
    {
        int pid = get_mpi_rank();
        WalkConfig walk_conf;
        if (!opt.output_path.empty())
        {
            corpus.push_back((opt.output_path + to_string(pid)));
            walk_conf.set_output_file((opt.output_path + to_string(pid)).c_str());
            cout << (opt.output_path + to_string(pid)).c_str() << endl;
        }
        if (opt.set_rate)
        {
            walk_conf.set_walk_rate(opt.rate);
        }
        Timer timer;
        printf("================= RANDOM WALK ================\n");
        graph.random_walk(&walker_conf, &tr_conf, &walk_conf);
        double sum_time = timer.duration();
        double walk_time = sum_time - graph.other_time;
        // printf("[p%u][sum time:]%lf [walk time:]%lf [other time:]%lf\n", graph.get_local_partition_id(), sum_time, walk_time, graph.other_time);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    printf("> [p%d RANDOM WALKING TIME:] %lf  msg_produce time: %lf\n",get_mpi_rank(), timer.duration(),graph.msg_produce_time);
    // if(get_mpi_rank() == 0){
    //     FILE* stream_log = fopen("stream.log","a");
    //     fprintf(stream_log,"%f,",timer.duration());
    //     fclose(stream_log);
    // }
    


    if(get_mpi_rank()==0){
        cout<<"============partion table=========="<<endl;
        for(int p=0;p<get_mpi_size();p++){
            cout<<"part: "<<p<<" "<<graph.vertex_partition_begin[p]<<" ~ "<<graph.vertex_partition_end[p]<<endl;
        }
    }

    // FILE* fcn = fopen((to_string(get_mpi_rank())+string("cn.txt")).c_str(),"w");
    // fprintf(fcn,"id      \tcn\n");
    // for(int i=0;i<graph.vertex_cn.size();i++){
    //     fprintf(fcn,"%-8d\t%-8d\n",i, graph.vertex_cn[i]);
    // }
    // fclose(fcn);

    MPI_Allreduce(MPI_IN_PLACE,graph.vertex_cn.data(), graph.get_vertex_num(), get_mpi_data_type<int>(), MPI_SUM, MPI_COMM_WORLD);

    // ================= annotation line ====================

    graph.new_sort.resize(graph.v_num);
    for(vertex_id_t v_i=0; v_i < graph.v_num;v_i++){
        graph.new_sort[v_i] = v_i;
    }

   
    if(0 == get_mpi_rank())
    {
      printf("[round times] %d\n",graph.round_count);
      for(int i=0;i<graph.round_length.size();i++)
      {
        printf("%zu ",graph.round_length[i]);
      }
      printf("\n increment \n");
      for(int i=1;i<graph.round_length.size();i++)
      {
          cout<<graph.round_length[i]-graph.round_length[i-1]<<" ";
      }


    }
    printf("================= EMBEDDING ================\n");

   
    // timer.restart();
    trainer_thread.join();
    // dsgl(argc, argv,&graph.vertex_cn,&graph.new_sort,&graph);
    // cuda_word2vec(argc,argv,&graph.vertex_cn,&graph.local_corpus);
    // printf("> [%d EMBBEDDING TIME:] %lf \n", get_mpi_rank(),timer.duration());
    printf("> [%d SUM TIME:] %lf \n", get_mpi_rank(),sum_timer.duration());
    return 0;
}
