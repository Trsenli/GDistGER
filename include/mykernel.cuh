#ifndef MYKERNEL_CUH
#define MYKERNEL_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "type.hpp"
#include  <mutex>
#include <vector>

using std::vector;
using std::mutex;


extern "C"  int cuda_word2vec (int argc, char **argv, vector<int>* vertex_cn, vector<int>*local_corpus);
extern "C" void trainer_caller(size_t c_start,size_t c_end);
extern "C" void add_corps2queue(vector<int> &tmp);
#endif