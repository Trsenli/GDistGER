#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "type.hpp"

using std::vector;


extern "C"  int cuda_word2vec (int argc, char **argv, vector<int>* vertex_cn, vector<int>*local_corpus);