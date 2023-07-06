#include <algorithm>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utility>

#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <xmmintrin.h>
// #include <sys/time.h>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include <map>
#include <mutex>
#include <queue>
#include <vector>

#include "mykernel.cuh"
#include "type.hpp"
#include "util.hpp"
// #include "mpi_helper.hpp"
// #include "walk.hpp"

using namespace std;
using std::vector;
extern mutex corpus_lock;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MPI_SCALAR MPI_FLOAT

#define MAX_SENTENCE 700
#define checkCUDAerr(err)                                       \
  {                                                             \
    cudaError_t cet = err;                                      \
    if (cudaSuccess != cet)                                     \
    {                                                           \
      printf("[ %d ]%s %d : %s\n", my_rank, __FILE__, __LINE__, \
             cudaGetErrorString(cet));                          \
      exit(0);                                                  \
    }                                                           \
  }

typedef float real;
typedef unsigned int uint;
typedef unsigned long long ulonglong;
typedef u_int32_t vertex_id_t;

vector<int> *cu_vertex_cn = nullptr;
vector<int> *cu_local_corpus = nullptr;

int num_procs = 1, my_rank = -1;
int message_size = 1024, min_sync_words = 1024, full_sync_times = 0;
real_t model_sync_period = 0.1f;

const int vocab_hash_size =
    30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word
{
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5,
    min_reduce = 1, reuseNeg = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 128;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0,
          classes = 0;
float alpha = 0.025, starting_alpha, sample = 1e-3;
float *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// ==== DIY ===========
vector<int> word_freq_block_ind;
struct vocab_vertex
{
  int cn;
  vertex_id_t id;
  vocab_vertex(){};
  vocab_vertex(uint _cn, vertex_id_t _id) : cn(_cn), id(_id){};
};
vector<vocab_vertex> v_vocab;
vector<vertex_id_t> hash2v_vocab;
double datamove_time = 0.0;
// ===================

// FOR CUDA
int *vocab_codelen, *vocab_point, *d_vocab_codelen, *d_vocab_point;
char *vocab_code, *d_vocab_code;
int *d_table;
float *d_syn0, *d_syn1, *d_expTable;

inline unsigned int getNumZeros(unsigned int v)
{
  unsigned int numzeros = 0;
  while (!(v & 0x1))
  {
    numzeros++;
    v = v >> 1;
  }
  return numzeros;
}
enum Trainer_flag
{
  TRAIN_ON,
  TRAIN_SUSPEND,
  TRAIN_OFF
};
mutex state_lock;
mutex queue_lock;
volatile Trainer_flag trainer_state = TRAIN_SUSPEND;
struct TrainTask
{
  size_t start_idx;
  size_t end_idx;
  vector<int> task_corpus;
  TrainTask(size_t si, size_t ei) : start_idx(si), end_idx(ei){};
  TrainTask(vector<int> &tmp)
  {
    task_corpus = std::move(tmp);
    start_idx = 0;
    end_idx = tmp.size();
  };
};
queue<TrainTask> train_queue;
// @brief: modifiy the state of trainer. And create the training task.
// when task finished, inform trainer to stop
// when task created, inform trainer on if it was suspending.
void trainer_caller(size_t c_start, size_t c_end)
{
  cout << "[ Trainer Caller ] invoked | " << c_start << "  " << c_end << endl;
  spdlog::debug("create TrainTask {0} {1}", c_start, c_end);
  queue_lock.lock();
  train_queue.push(TrainTask(c_start, c_end));
  queue_lock.unlock();
}
void add_corps2queue(vector<int> &tmp)
{
  spdlog::debug("add corpus to queue.Size {0}", tmp.size());
  lock_guard<mutex> lg(queue_lock);
  train_queue.push(TrainTask(tmp));
}
__device__ float reduceInWarp(float f)
{
  for (int i = warpSize / 2; i > 0; i /= 2)
  {
    f += __shfl_xor(f, i, 32);
  }
  return f;
}

__device__ void warpReduce(volatile float *sdata, int tid)
{
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}
template <unsigned int VSIZE>
__global__ void
__old_sgNegReuse(const int window, const int layer1_size, const int negative,
                 const int vocab_size, float alpha, const int *__restrict__ sen,
                 const int *__restrict__ sentence_length, float *syn1,
                 float *syn0, const int *negSample)
{
  __shared__ float neu1e[VSIZE];
  const int sentIdx_s = sentence_length[blockIdx.x];
  const int sentIdx_e = sentence_length[blockIdx.x + 1];
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int dxy = blockDim.x * blockDim.y;

  int _negSample;
  if (threadIdx.y < negative)
  { // Get the negative sample
    _negSample = negSample[blockIdx.x * negative + threadIdx.y];
  }

  for (int sentPos = sentIdx_s; sentPos < sentIdx_e; sentPos++) {
    int word = sen[sentPos]; // Target word
    if (word == -1)
      continue;

    for (int a = 0; a < window * 2 + 1; a++)
      if (a != window)
      {
        int c = sentPos - window + a; // The index of context word
        if (c >= sentIdx_s && c < sentIdx_e && sen[c] != -1)
        {
          int l1 = sen[c] * layer1_size;

          for (int i = tid; i < layer1_size; i += dxy)
          {
            neu1e[i] = 0;
          }
          __syncthreads();

          int target, label, l2;
          float f = 0, g;
          if (threadIdx.y == negative)
          { // Positive sample
            target = word;
            label = 1;
          }
          else
          { // Negative samples
            if (_negSample == word)
              goto NEGOUT;
            target = _negSample;
            label = 0;
          }
          l2 = target * layer1_size;

          for (int i = threadIdx.x; i < layer1_size;
               i += blockDim.x)
          { // Get gradient
            f += syn0[i + l1] * syn1[i + l2];
          }
          f = reduceInWarp(f);
          if (f > MAX_EXP)
            g = (label - 1) * alpha;
          else if (f < -MAX_EXP)
            g = (label - 0) * alpha;
          else
          {
            int tInt = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
            float t = exp((tInt / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
            t = t / (t + 1);
            g = (label - t) * alpha;
          }

          for (int i = threadIdx.x; i < layer1_size; i += warpSize)
          {
            atomicAdd(&neu1e[i], g * syn1[i + l2]);
          }
          for (int i = threadIdx.x; i < layer1_size;
               i += warpSize)
          { // Update syn1 of negative sample
            syn1[i + l2] += g * syn0[i + l1];
          }

        NEGOUT:
          __syncthreads();

          for (int i = tid; i < layer1_size;
               i += dxy)
          { // Update syn0 of context word
            atomicAdd(&syn0[i + l1], neu1e[i]);
          }
        }
      }
  }
}

template <unsigned int VSIZE> // 向量size
__global__ void __sgNegReuse(const int window, const int layer1_size,
                             const int negative, const int vocab_size,
                             float alpha, const int *__restrict__ sen,
                             const int *__restrict__ sentence_length,
                             float *syn1, float *syn0, const int *negSample)
{
  __shared__ float neu1e[VSIZE]; // 线程块内的共享内存

  // sentence_length eg: {2,6,19,33,45 ...}
  // const int sentIdx_s = sentence_length[blockIdx.x];// start
  // const int sentIdx_e = sentence_length[blockIdx.x + 1]; // end
  const int sentIdx_s = sentence_length[blockIdx.x * 2];
  int senEnd = -1;
  if (blockIdx.x + 2 > gridDim.x)
    senEnd = gridDim.x;
  else
    senEnd = blockIdx.x + 2;
  const int sentIdx_e = sentence_length[senEnd];

  const int sentIdx_m = sentence_length[blockIdx.x + 1];

  const int tid =
      threadIdx.x + blockDim.x * threadIdx.y; // thread ID (in global)
  const int dxy =
      blockDim.x *
      blockDim.y; // block 里面有多少线程，一个负样本的计算能给32个线程
  // ========= Load Buffer ==============
  // 一个block 对应两个句子

  const int MAX_SEN_LENGTH = sentIdx_e - sentIdx_s;
  // __shared__ float  negbuf[VSIZE*(NEGNUM + MAX_SEN_LENGTH *2) ];
  // __shared__ float  negbuf[VSIZE*NEGNUM];
  __shared__ float *negbuf;
  // __shared__ float  senbuf[VSIZE*MAX_SEN_LENGTH * 2];
  __shared__ float *senbuf;

  __shared__ int *negTable;
  // __shared__ int negTable[NEGNUM + MAX_SEN_LENGTH *2];
  __shared__ int *senTable;

  if (tid == 0)
  {
    senbuf = new float[VSIZE * MAX_SEN_LENGTH];
    senTable = new int[MAX_SEN_LENGTH];
  }
  else if (tid == 1)
  {
    negbuf = new float[VSIZE * MAX_SEN_LENGTH + negative];
    negTable = new int[negative + MAX_SEN_LENGTH];
  }

  if (tid == 2)
  {
    for (int i = 0; i < MAX_SEN_LENGTH; i++)
    {
      senTable[i] = -1;
    }
    for (int i = 0; i < negative + MAX_SEN_LENGTH; i++)
    {
      negTable[i] = -1;
    }
  }

  __syncthreads();

  // load sentence buffer from syn0 and syn1
  for (int sentPos = sentIdx_s; sentPos < sentIdx_e;
       sentPos++)
  { // 遍历句子中的词
    int word = sen[sentPos];
    senTable[sentPos - sentIdx_s] = word; // 记录buffer 每个位置是哪个词
    negTable[sentPos - sentIdx_s] = word;
    if (word == -1)
      continue;
    uint64_t lsrc = word * layer1_size;
    uint64_t ldst = (sentPos - sentIdx_s) * layer1_size;
    for (int i = tid; i < layer1_size; i += dxy)
    {
      senbuf[ldst + i] = syn0[lsrc + i];
      negbuf[ldst + i] = syn1[lsrc + i];
    }
  }

  // load negative buffer
  for (int n = 0; n < negative; n++)
  {
    int neg = negSample[blockIdx.x * negative + n];
    negTable[n + MAX_SEN_LENGTH * 2] = neg;
    uint64_t lsrc = neg * layer1_size;
    uint64_t ldst = (n + MAX_SEN_LENGTH * 2) * layer1_size;
    for (int i = tid; i < layer1_size; i += dxy)
    {
      negbuf[ldst + i] = syn1[lsrc + i];
    }
  }

  __syncthreads();

  auto getNegBufIdx = [&](int neg) -> int
  {
    for (int n = 0; n < negative + MAX_SEN_LENGTH; n++)
    {
      if (negTable[n] == neg)
        return n;
    }
    return -1;
  };

  auto getSenBufIdx = [&](int word) -> int
  {
    for (int s = 0; s < MAX_SEN_LENGTH; s++)
    {
      if (senTable[s] == word)
        return s;
    }
    return -1;
  };

  // ========== TRAIN ============

  // negSample cnt_sentence * negative
  int _negSample; // 一个y对应一个 负样本。 一个句子用一组负样本
  if (threadIdx.y < negative)
  { // Get the negative sample
    _negSample = negSample[blockIdx.x * negative + threadIdx.y];
  }

  int tmp_sent_s = sentIdx_s;
  int tmp_sent_e = sentIdx_m;
  for (int sentPos = sentIdx_s; sentPos < sentIdx_e; sentPos++) // 遍历句子
  {                                                             // 遍历句子中的词

    if (sentPos >= sentIdx_m)
    {
      tmp_sent_s = sentIdx_m;
      tmp_sent_e = sentIdx_e;
    }
    int word = sen[sentPos]; // Target word
    if (word == -1)
      continue;

    for (int a = 0; a < window * 2 + 1; a++) // 遍历窗口
      if (a != window)
      {                               // 对于每个 context word
        int c = sentPos - window + a; // The index of context word
        if (c >= tmp_sent_s && c < tmp_sent_e && sen[c] != -1)
        {
          // int l1 = sen[c] * layer1_size; // 定位 上下文词 的向量
          uint64_t l1 = getSenBufIdx(c) * layer1_size;

          for (int i = tid; i < layer1_size; i += dxy)
          {
            neu1e[i] = 0;
          }
          __syncthreads();

          int target, label, real_l2;
          uint64_t l2;
          float f = 0, g;
          if (threadIdx.y == negative)
          { // Positive sample
            target = word;
            label = 1;
          }
          else
          { // Negative samples
            if (_negSample == word)
              goto NEGOUT; // 负样本和中心词取一样了，跳过
            target = _negSample;
            label = 0;
          }
          real_l2 = target * layer1_size;
          // 定位target 的向量
          // l2 = target * layer1_size; // 对于一个上下文词来说，target
          // 要么是中心词就是正样本 ，如果是负样本词，就是负样本。
          l2 = getNegBufIdx(target) * layer1_size;

          // 每个上下文词与负样本做运算
          // 梯度计算

          for (int i = threadIdx.x; i < layer1_size;
               i += blockDim.x)
          { // Get gradient
            // f += syn0[i + l1] * syn1[i + l2];
            f += senbuf[i + l1] * negbuf[i + l2];
          }
          f = reduceInWarp(f);
          if (f > MAX_EXP)
            g = (label - 1) * alpha;
          else if (f < -MAX_EXP)
            g = (label - 0) * alpha;
          else
          {
            int tInt = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
            float t = exp((tInt / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
            t = t / (t + 1);
            g = (label - t) * alpha;
          }

          for (int i = threadIdx.x; i < layer1_size; i += warpSize)
          {
            // atomicAdd(&neu1e[i], g * syn1[i + l2]);
            atomicAdd(&neu1e[i], g * negbuf[i + l2]);
          }

          for (int i = threadIdx.x; i < layer1_size;
               i += warpSize)
          { // Update syn1 of negative sample
            // syn1[i + l2] += g * syn0[i + l1];
            syn1[i + real_l2] += g * senbuf[i + l1];
          }

        NEGOUT:
          __syncthreads();

          int real_l1 = sen[c] * layer1_size;
          for (int i = tid; i < layer1_size;
               i += dxy)
          { // Update syn0 of context word
            // atomicAdd(&syn0[i + l1], neu1e[i]);
            atomicAdd(&syn0[i + real_l1], neu1e[i]);
          }
        }
      }
  }
  if (tid == 0)
  {
    delete senbuf;
    delete negbuf;
    delete senTable;
    delete negTable;
  }
  // cudaFree(senbuf);
}

template <unsigned int FSIZE>
__global__ void skip_gram_kernel(
    int window, int layer1_size, int negative, int hs, int table_size,
    int vocab_size, float alpha, const float *__restrict__ expTable,
    const int *__restrict__ table, const int *__restrict__ vocab_codelen,
    const int *__restrict__ vocab_point, const char *__restrict__ vocab_code,
    const int *__restrict__ sen, const int *__restrict__ sentence_length,
    float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1];
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size)
    for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e;
         sentence_position++)
    {
      int word = sen[sentence_position];
      if (word == -1)
        continue;
      float neu1e = 0;
      next_random = next_random * (unsigned long)2514903917 + 11;
      int b = next_random % window;

      for (int a = b; a < window * 2 + 1 - b; a++)
        if (a != window)
        {
          int c = sentence_position - window + a;
          if (c < sent_idx_s)
            continue;
          if (c >= sent_idx_e)
            continue;
          int last_word = sen[c];
          if (last_word == -1)
            continue;
          int l1 = last_word * layer1_size;
          neu1e = 0;

          // HIERARCHICAL SOFTMAX
          if (hs)
            for (int d = vocab_codelen[word]; d < vocab_codelen[word + 1];
                 d++)
            {
              int l2 = vocab_point[d] * layer1_size;

              if (threadIdx.x < FSIZE)
                f[threadIdx.x] =
                    syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
              __syncthreads();
              if (threadIdx.x >= FSIZE)
                f[threadIdx.x % (FSIZE)] +=
                    syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
              __syncthreads();
              for (int i = (FSIZE / 2); i > 0; i /= 2)
              {
                if (threadIdx.x < i)
                  f[threadIdx.x] += f[i + threadIdx.x];
                __syncthreads();
              }

              if (f[0] <= -MAX_EXP)
                continue;
              else if (f[0] >= MAX_EXP)
                continue;
              else if (threadIdx.x == 0)
              {
                f[0] = expTable[(int)((f[0] + MAX_EXP) *
                                      (EXP_TABLE_SIZE / MAX_EXP / 2))];
                g = (1 - vocab_code[d] - f[0]) * alpha;
              }
              __syncthreads();

              neu1e += g * syn1[threadIdx.x + l2];
              atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
            }

          // NEGATIVE SAMPLING
          if (negative > 0)
            for (int d = 0; d < negative + 1; d++)
            {
              int target, label;
              if (d == 0)
              {
                target = word;
                label = 1;
              }
              else
              {
                next_random = next_random * (unsigned long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0)
                  target = next_random % (vocab_size - 1) + 1;
                if (target == word)
                  continue;
                label = 0;
              }
              int l2 = target * layer1_size;

              if (threadIdx.x < FSIZE)
                f[threadIdx.x] =
                    syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
              __syncthreads();
              if (threadIdx.x >= FSIZE)
                f[threadIdx.x % (FSIZE)] +=
                    syn0[threadIdx.x + l1] * syn1[threadIdx.x + l2];
              __syncthreads();
              for (int i = (FSIZE / 2); i > 0; i /= 2)
              {
                if (threadIdx.x < i)
                  f[threadIdx.x] += f[i + threadIdx.x];
                __syncthreads();
              }
              if (threadIdx.x == 0)
              {
                if (f[0] > MAX_EXP)
                  g = (label - 1) * alpha;
                else if (f[0] < -MAX_EXP)
                  g = (label - 0) * alpha;
                else
                  g = (label -
                       expTable[(int)((f[0] + MAX_EXP) *
                                      (EXP_TABLE_SIZE / MAX_EXP / 2))]) *
                      alpha;
              }
              __syncthreads();

              neu1e += g * syn1[threadIdx.x + l2];
              atomicAdd(&syn1[threadIdx.x + l2], g * syn0[threadIdx.x + l1]);
            }

          atomicAdd(&syn0[threadIdx.x + l1], neu1e);
        }
    }
}

template <unsigned int FSIZE>
__global__ void
cbow_kernel(int window, int layer1_size, int negative, int hs, int table_size,
            int vocab_size, float alpha, const float *__restrict__ expTable,
            const int *__restrict__ table,
            const int *__restrict__ vocab_codelen,
            const int *__restrict__ vocab_point,
            const char *__restrict__ vocab_code, const int *__restrict__ sen,
            const int *__restrict__ sentence_length, float *syn1, float *syn0)
{
  __shared__ float f[FSIZE], g;

  int sent_idx_s = sentence_length[blockIdx.x];
  int sent_idx_e = sentence_length[blockIdx.x + 1];
  unsigned long next_random = blockIdx.x;

  if (threadIdx.x < layer1_size)
    for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e;
         sentence_position++)
    {
      int word = sen[sentence_position];
      if (word == -1)
        continue;
      float neu1 = 0;
      float neu1e = 0;
      next_random = next_random * (unsigned long)2514903917 + 11;
      int b = next_random % window;

      int cw = 0;
      for (int a = b; a < window * 2 + 1 - b; a++)
        if (a != window)
        {
          int c = sentence_position - window + a;
          if (c < sent_idx_s)
            continue;
          if (c >= sent_idx_e)
            continue;
          int last_word = sen[c];
          if (last_word == -1)
            continue;
          neu1 += syn0[last_word * layer1_size + threadIdx.x];
          cw++;
        }

      if (cw)
      {
        neu1 /= cw;

        // HIERARCHICAL SOFTMAX
        if (hs)
          for (int d = vocab_codelen[word]; d < vocab_codelen[word + 1]; d++)
          {
            int l2 = vocab_point[d] * layer1_size;

            if (threadIdx.x < FSIZE)
              f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
            __syncthreads();
            if (threadIdx.x >= FSIZE)
              f[threadIdx.x % (FSIZE)] += neu1 * syn1[threadIdx.x + l2];
            __syncthreads();
            for (int i = (FSIZE / 2); i > 0; i /= 2)
            {
              if (threadIdx.x < i)
                f[threadIdx.x] += f[i + threadIdx.x];
              __syncthreads();
            }

            if (f[0] <= -MAX_EXP)
              continue;
            else if (f[0] >= MAX_EXP)
              continue;
            else if (threadIdx.x == 0)
            {
              f[0] = expTable[(int)((f[0] + MAX_EXP) *
                                    (EXP_TABLE_SIZE / MAX_EXP / 2))];
              g = (1 - vocab_code[d] - f[0]) * alpha;
            }
            __syncthreads();

            neu1e += g * syn1[threadIdx.x + l2];
            atomicAdd(&syn1[threadIdx.x + l2], g * neu1);
          }

        // NEGATIVE SAMPLING
        if (negative > 0)
          for (int d = 0; d < negative + 1; d++)
          {
            int target, label;
            if (d == 0)
            {
              target = word;
              label = 1;
            }
            else
            {
              next_random = next_random * (unsigned long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0)
                target = next_random % (vocab_size - 1) + 1;
              if (target == word)
                continue;
              label = 0;
            }
            int l2 = target * layer1_size;

            if (threadIdx.x < FSIZE)
              f[threadIdx.x] = neu1 * syn1[threadIdx.x + l2];
            __syncthreads();
            if (threadIdx.x >= FSIZE)
              f[threadIdx.x % (FSIZE)] += neu1 * syn1[threadIdx.x + l2];
            __syncthreads();
            for (int i = (FSIZE / 2); i > 0; i /= 2)
            {
              if (threadIdx.x < i)
                f[threadIdx.x] += f[i + threadIdx.x];
              __syncthreads();
            }
            if (threadIdx.x == 0)
            {
              if (f[0] > MAX_EXP)
                g = (label - 1) * alpha;
              else if (f[0] < -MAX_EXP)
                g = (label - 0) * alpha;
              else
                g = (label - expTable[(int)((f[0] + MAX_EXP) *
                                            (EXP_TABLE_SIZE / MAX_EXP / 2))]) *
                    alpha;
            }
            __syncthreads();

            neu1e += g * syn1[l2 + threadIdx.x];
            atomicAdd(&syn1[l2 + threadIdx.x], g * neu1);
          }

        for (int a = b; a < window * 2 + 1 - b; a++)
          if (a != window)
          {
            int c = sentence_position - window + a;
            if (c < sent_idx_s)
              continue;
            if (c >= sent_idx_e)
              continue;
            int last_word = sen[c];
            if (last_word == -1)
              continue;
            atomicAdd(&syn0[last_word * layer1_size + threadIdx.x], neu1e);
          }
      }
    }
}

void InitVocabStructCUDA()
{
  vocab_codelen = (int *)malloc((vocab_size + 1) * sizeof(int));
  vocab_codelen[0] = 0;
  for (int i = 1; i < vocab_size + 1; i++)
    vocab_codelen[i] = vocab_codelen[i - 1] + vocab[i - 1].codelen;
  vocab_point = (int *)malloc(vocab_codelen[vocab_size] * sizeof(int));
  vocab_code = (char *)malloc(vocab_codelen[vocab_size] * sizeof(char));

  checkCUDAerr(
      cudaMalloc((void **)&d_vocab_codelen, (vocab_size + 1) * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_point,
                          vocab_codelen[vocab_size] * sizeof(int)));
  checkCUDAerr(cudaMalloc((void **)&d_vocab_code,
                          vocab_codelen[vocab_size] * sizeof(char)));

  for (int i = 0; i < vocab_size; i++)
  {
    for (int j = 0; j < vocab[i].codelen; j++)
    {
      vocab_code[vocab_codelen[i] + j] = vocab[i].code[j];
      vocab_point[vocab_codelen[i] + j] = vocab[i].point[j];
    }
  }

  checkCUDAerr(cudaMemcpy(d_vocab_codelen, vocab_codelen,
                          (vocab_size + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_point, vocab_point,
                          vocab_codelen[vocab_size] * sizeof(int),
                          cudaMemcpyHostToDevice));
  checkCUDAerr(cudaMemcpy(d_vocab_code, vocab_code,
                          vocab_codelen[vocab_size] * sizeof(char),
                          cudaMemcpyHostToDevice));
}

void InitUnigramTable()
{ // 提前生成负样本的采样表
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++)
    train_words_pow += pow(v_vocab[a].cn, power);
  i = 0;
  d1 = pow(v_vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++)
  {
    table[a] = i;
    if (a / (double)table_size > d1)
    {
      i++;
      d1 += pow(v_vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size)
      i = vocab_size - 1;
  }
  // FOR CUDA
  checkCUDAerr(
      cudaMalloc((void **)&d_table, table_size * sizeof(int))); // 拷贝到GPU里
  checkCUDAerr(cudaMemcpy(d_table, table, table_size * sizeof(int),
                          cudaMemcpyHostToDevice));
}

// Reads a single word from a file, assuming space + tab + EOL to be word
// boundaries
void ReadWord(char *word, FILE *fin)
{
  int a = 0, ch;
  while (!feof(fin))
  {
    ch = fgetc(fin);
    if (ch == 13)
      continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
    {
      if (a > 0)
      {
        if (ch == '\n')
          ungetc(ch, fin);
        break;
      }
      if (ch == '\n')
      {
        strcpy(word, (char *)"</s>");
        return;
      }
      else
        continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1)
      a--; // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word)
{
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++)
    hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found,
// returns -1
int SearchVocab(char *word)
{
  unsigned int hash = GetWordHash(word);
  while (1)
  {
    if (vocab_hash[hash] == -1)
      return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word))
      return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  //  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin)
{
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin))
    return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word)
{
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING)
    length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size)
  {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size *
                                                    sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1)
    hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b)
{
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab()
{
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++)
  {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0))
    {
      vocab_size--;
      free(vocab[a].word);
    }
    else
    {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) *
                                                  sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++)
  {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab()
{
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce)
    {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    }
    else
      free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++)
  {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1)
      hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree()
{
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary =
      (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node =
      (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++)
    count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++)
    count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a
  // time
  for (a = 0; a < vocab_size - 1; a++)
  {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0)
    {
      if (count[pos1] < count[pos2])
      {
        min1i = pos1;
        pos1--;
      }
      else
      {
        min1i = pos2;
        pos2++;
      }
    }
    else
    {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0)
    {
      if (count[pos1] < count[pos2])
      {
        min2i = pos1;
        pos1--;
      }
      else
      {
        min2i = pos2;
        pos2++;
      }
    }
    else
    {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++)
  {
    b = a;
    i = 0;
    while (1)
    {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2)
        break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++)
    {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void ConstVobcabFromMemory()
{
  vocab_size = cu_vertex_cn->size();
  v_vocab.resize(vocab_size);
  for (int i = 0; i < v_vocab.size(); i++)
  {
    v_vocab[i].cn = (*cu_vertex_cn)[i];
    v_vocab[i].id = i;
  }

  sort(v_vocab.begin(), v_vocab.end(),
       [](vocab_vertex a, vocab_vertex b)
       { return a.cn > b.cn; });

  train_words = 0;
  for (int i = 0; i < cu_vertex_cn->size(); i++)
  {
    train_words += (*cu_vertex_cn)[i];
  }

  hash2v_vocab.resize(vocab_size);
  for (int i = 0; i < vocab_size; i++)
  {
    hash2v_vocab[v_vocab[i].id] = i;
  }
  if (my_rank == 0 && debug_mode > 0)
  {
    printf("vertex Vocab size: %lld\n", vocab_size);
    printf("train_words: %lld\n", train_words);
  }
}

void LearnVocabFromTrainFile()
{
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL)
  {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1)
  {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0))
    {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1)
    {
      // a = AddWordToVocab(word);
      // vocab[a].cn = 1; // 词存在 vocab 里面
    }
    else
      vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7)
      ReduceVocab();
  }
  // SortVocab();
  if (debug_mode > 0)
  {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab()
{
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++)
    fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab()
{
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL)
  {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  vocab_size = 0;
  while (1)
  {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0)
  {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL)
  {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet()
{
  long long a, b;
  unsigned long long next_random = 1;
  // a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size
  // * sizeof(float));
  checkCUDAerr(cudaHostAlloc(
      (void **)&syn0, (long long)vocab_size * layer1_size * sizeof(float),
      cudaHostAllocWriteCombined | cudaHostAllocMapped));
  if (syn0 == NULL)
  {
    printf("Memory allocation failed\n");
    exit(1);
  }
  if (hs)
  {
    a = posix_memalign((void **)&syn1, 128,
                       (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL)
    {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0;
    checkCUDAerr(cudaMalloc((void **)&d_syn1, (long long)vocab_size *
                                                  layer1_size * sizeof(float)));
    checkCUDAerr(cudaMemcpy(d_syn1, syn1,
                            (long long)vocab_size * layer1_size * sizeof(float),
                            cudaMemcpyHostToDevice));
  }
  if (negative > 0)
  { // ==== 使用负采样  syn1neg 应该对应的 woh
    // a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size *
    // layer1_size * sizeof(float));
    checkCUDAerr(cudaHostAlloc(
        (void **)&syn1neg, (long long)vocab_size * layer1_size * sizeof(float),
        cudaHostAllocWriteCombined | cudaHostAllocMapped));
    if (syn1neg == NULL)
    {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
    checkCUDAerr(
        cudaMalloc((void **)&d_syn1,
                   (long long)vocab_size * layer1_size *
                       sizeof(float))); // d_syn1 对应的 syn1，这里是 syn1neg
    checkCUDAerr(cudaMemcpy(d_syn1, syn1neg,
                            (long long)vocab_size * layer1_size * sizeof(float),
                            cudaMemcpyHostToDevice));
    // checkCUDAerr(cudaHostGetDevicePointer(&d_syn1, syn1neg, 0));
  }
  for (a = 0; a < vocab_size; a++)
    for (b = 0; b < layer1_size; b++)
    { // 初始化随机值
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0[a * layer1_size + b] =
          (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
  checkCUDAerr(cudaMalloc((void **)&d_syn0,
                          (long long)vocab_size * layer1_size * sizeof(float)));
  checkCUDAerr(cudaMemcpy(d_syn0, syn0,
                          (long long)vocab_size * layer1_size * sizeof(float),
                          cudaMemcpyHostToDevice));
  // checkCUDAerr(cudaHostGetDevicePointer(&d_syn0, syn0, 0));
  // CreateBinaryTree();
}

void cbowKernel(int *d_sen, int *d_sent_len, float alpha, int cnt_sentence,
                int reduSize)
{
  int bDim = layer1_size;
  int gDim = cnt_sentence;
  switch (reduSize)
  {
  case 128:
    cbow_kernel<64><<<gDim, bDim>>>(
        window, layer1_size, negative, hs, table_size, vocab_size, alpha,
        d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
        d_sen, d_sent_len, d_syn1, d_syn0);
    break;
  case 256:
    cbow_kernel<128><<<gDim, bDim>>>(
        window, layer1_size, negative, hs, table_size, vocab_size, alpha,
        d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
        d_sen, d_sent_len, d_syn1, d_syn0);
    break;
  case 512:
    cbow_kernel<256><<<gDim, bDim>>>(
        window, layer1_size, negative, hs, table_size, vocab_size, alpha,
        d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
        d_sen, d_sent_len, d_syn1, d_syn0);
    break;
  default:
    printf("Can't support on vector size = %lld\n", layer1_size);
    exit(1);
    break;
  }
}

void sgKernel(int *d_sen, int *d_sent_len, int *d_negSample, float alpha,
              int cnt_sentence, int reduSize)
{
  int bDim = layer1_size; // block size 一个block 里有layer1_size 个 threads
  // int gDim= cnt_sentence; // grid size 一个grid 里有 cnt_sentence 个block
  int gDim = cnt_sentence / 2 + cnt_sentence % 2 ? 1 : 0;

  if (reuseNeg)
  {                                    // A sentence share negative samples
    dim3 bDimNeg(32, negative + 1, 1); // block 维度
    // 双句共享负样本
    // dim3 bDimNeg(32,negative+2,1);
    switch (layer1_size)
    {
    case 128:
      __old_sgNegReuse<128>
          <<<gDim, bDimNeg>>>(window, layer1_size, negative, vocab_size, alpha,
                              d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
      break;
    case 64:
      __sgNegReuse<64><<<gDim, bDimNeg>>>(window, layer1_size, negative,
                                          vocab_size, alpha, d_sen, d_sent_len,
                                          d_syn1, d_syn0, d_negSample);
      break;
    case 200:
      __sgNegReuse<200><<<gDim, bDimNeg>>>(window, layer1_size, negative,
                                           vocab_size, alpha, d_sen, d_sent_len,
                                           d_syn1, d_syn0, d_negSample);
      break;
    case 300:
      __old_sgNegReuse<300>
          <<<gDim, bDimNeg>>>(window, layer1_size, negative, vocab_size, alpha,
                              d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
      // __sgNegReuse<300><<<gDim, bDimNeg>>>(window, layer1_size, negative,
      // vocab_size, alpha, d_sen, d_sent_len, d_syn1, d_syn0, d_negSample);
      break;
    default:
      printf("Can't support on vector size = %lld\n", layer1_size);
      exit(1);
      break;
    }
  }
  else
  {
    switch (reduSize)
    {
    case 128:
      skip_gram_kernel<64><<<gDim, bDim>>>(
          window, layer1_size, negative, hs, table_size, vocab_size, alpha,
          d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
          d_sen, d_sent_len, d_syn1, d_syn0);
      break;
    case 256:
      skip_gram_kernel<128><<<gDim, bDim>>>(
          window, layer1_size, negative, hs, table_size, vocab_size, alpha,
          d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
          d_sen, d_sent_len, d_syn1, d_syn0);
      break;
    case 512:
      skip_gram_kernel<256><<<gDim, bDim>>>(
          window, layer1_size, negative, hs, table_size, vocab_size, alpha,
          d_expTable, d_table, d_vocab_codelen, d_vocab_point, d_vocab_code,
          d_sen, d_sent_len, d_syn1, d_syn0);
      break;
    default:
      printf("Can't support on vector size = %lld\n", layer1_size);
      exit(1);
      break;
    }
  }
}

int num_syncs = 0;
int val_sync_vocab_size = 0;
void sync_CPU_with_GPU()
{
  num_syncs += MAX_SENTENCE * 10;
  int sync_chunk_size = message_size * 1024 * 1024 / (layer1_size * 4);
  // int sync_vocab_size =  min((long long)(1 << getNumZeros(num_syncs)) *
  // min_sync_words, vocab_size);
  int num_rounds = val_sync_vocab_size / sync_chunk_size +
                   ((val_sync_vocab_size % sync_chunk_size > 0) ? 1 : 0);
  for (int r = 0; r < num_rounds; r++)
  {
    int start = r * sync_chunk_size;
    int sync_size = min(sync_chunk_size, val_sync_vocab_size - start);
    checkCUDAerr(cudaMemcpy(syn0 + start * layer1_size,
                            d_syn0 + start * layer1_size,
                            sync_size * layer1_size, cudaMemcpyDeviceToHost));
    checkCUDAerr(cudaMemcpy(syn1neg + start * layer1_size,
                            d_syn1 + start * layer1_size,
                            sync_size * layer1_size, cudaMemcpyDeviceToHost));
  }
  cudaDeviceSynchronize();
}

void word_freq_block() // Ma
{

  word_freq_block_ind.push_back(0);
  int j = 1;
  for (int i = 1; i < v_vocab.size(); i++)
  {
    //   // printf("freq: %d %d \n",i, v_vocab[i].cn);
    if (v_vocab[i].cn != v_vocab[i - 1].cn)
    {
      word_freq_block_ind.push_back(i);
      // printf("freq: %d %d %d \n",i, j, v_vocab[i].cn);
      j++;
    }
  }
  printf("freq_block num: %d \n", j);
}

__global__ void Test(int *a, int sz)
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int txy = blockDim.x * blockDim.y;
  for (int i = tid; i < sz; i += txy)
  {
    a[i] += 0;
  }
  __syncthreads();
  if (tid == 0)
    printf("Test ok\n");
}
void sync_GPU_with_CPU(unsigned int mpi_num_syncs)
{
  int sync_chunk_size = message_size * 1024 * 1024 / (layer1_size * 4);
  int sync_vocab_size =
      min((long long)(1 << getNumZeros(mpi_num_syncs)) * min_sync_words,
          vocab_size);
  int num_rounds = sync_vocab_size / sync_chunk_size +
                   ((sync_vocab_size % sync_chunk_size > 0) ? 1 : 0);
  for (int r = 0; r < num_rounds; r++)
  {
    int start = r * sync_chunk_size;
    int sync_size = min(sync_chunk_size, sync_vocab_size - start);
    checkCUDAerr(cudaMemcpy(syn0 + start * layer1_size,
                            d_syn0 + start * layer1_size,
                            sync_size * layer1_size, cudaMemcpyDeviceToHost));
    checkCUDAerr(cudaMemcpy(syn1neg + start * layer1_size,
                            d_syn1 + start * layer1_size,
                            sync_size * layer1_size, cudaMemcpyDeviceToHost));
  }
  cudaDeviceSynchronize();
}

void TrainModelThread()
{

  volatile bool compute_go = true;
  volatile int ready_threads = 0;
  int active_threads = 1;
  int num_threads = 2;

#pragma omp parallel num_threads(num_threads)
  {
    int id = omp_get_thread_num();

    if (id == 0)
    {
      int active_processes = 1;
      int active_processes_global = num_procs;
      ulonglong word_count_actual_global = 0;
      int sync_chunk_size = message_size * 1024 * 1024 / (layer1_size * 4);
      int full_sync_count = 1;
      unsigned int num_syncs = 0;

      while (ready_threads < num_threads - 1)
      {
        usleep(1);
      }
      MPI_Barrier(MPI_COMM_WORLD);

#pragma omp atomic
      ready_threads++;

      double start = omp_get_wtime();
      double sync_start = start;
      int sync_block_size = 0;

      while (1)
      {
        double sync_eclipsed = omp_get_wtime() - sync_start;
        if (sync_eclipsed > model_sync_period)
        {
          compute_go = false;
          num_syncs++;
          active_processes = (active_threads > 0 ? 1 : 0);

          // printf("sa\n");
          // synchronize parameters
          MPI_Allreduce(&active_processes, &active_processes_global, 1, MPI_INT,
                        MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(&word_count_actual, &word_count_actual_global, 1,
                        MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

          // determine if full sync
          int sync_vocab_size =
              min((long long)(1 << getNumZeros(num_syncs)) * min_sync_words,
                  vocab_size);
          real progress =
              word_count_actual_global / (real)(iter * train_words + 1);
          if ((full_sync_times > 0) &&
              (progress >
               (real)full_sync_count / (full_sync_times + 1) + 0.01f))
          {
            full_sync_count++;
            sync_vocab_size = vocab_size;
            int num_rounds = sync_vocab_size / sync_chunk_size +
                             ((sync_vocab_size % sync_chunk_size > 0) ? 1 : 0);
            for (int r = 0; r < num_rounds; r++)
            {
              int start = r * sync_chunk_size;
              int sync_size = min(sync_chunk_size, sync_vocab_size - start);

              MPI_Allreduce(MPI_IN_PLACE, syn0 + start * layer1_size,
                            sync_size * layer1_size, MPI_SCALAR, MPI_SUM,
                            MPI_COMM_WORLD);
              MPI_Allreduce(MPI_IN_PLACE, syn1neg + start * layer1_size,
                            sync_size * layer1_size, MPI_SCALAR, MPI_SUM,
                            MPI_COMM_WORLD);
            }
// printf("svs:%d  vs:%ld \n",sync_vocab_size,vocab_size);
// printf("num_proc: %d\n",num_procs);
#pragma simd
#pragma vector aligned
            for (int i = 0; i < sync_vocab_size * layer1_size; i++)
            {
              syn0[i] /= num_procs;
              syn1neg[i] /= num_procs;
            }
          }
          else
          {
            real *sync_block_in = (real *)_mm_malloc(
                (size_t)vocab_size * layer1_size * sizeof(real), 64);
            real *sync_block_out = (real *)_mm_malloc(
                (size_t)vocab_size * layer1_size * sizeof(real), 64);
            vector<int> sync_ind;

            if (my_rank == 0)
            {
              int i = 1;
              for (int j = 1; j < word_freq_block_ind.size(); j += 1)
              {
                int i = 1;
                for (int t = 0; t < 1; t++)
                {
                  int tem_ind = rand() % (word_freq_block_ind[j] -
                                          word_freq_block_ind[j - 1]) +
                                word_freq_block_ind[j - 1];
                  if (tem_ind == 0)
                  {
                    sync_ind.push_back(tem_ind);
                  }
                  else
                  {
                    if (tem_ind != sync_ind[i - 1])
                    {
                      sync_ind.push_back(tem_ind);
                      i++;
                    }
                  }
                }
              }
            }
            sync_block_size = sync_ind.size();

            MPI_Bcast(&sync_block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (my_rank != 0)
            {
              sync_ind.resize(sync_block_size);
            }
            MPI_Bcast(sync_ind.data(), sync_block_size, MPI_INT, 0,
                      MPI_COMM_WORLD);

            for (size_t i = 0; i < sync_block_size; i++)
            {
              // printf("id %d, sync_id %d\n",i, sync_ind[i]);

              memcpy(sync_block_in + i * layer1_size,
                     syn0 + (size_t)sync_ind[i] * layer1_size,
                     layer1_size * sizeof(real));
              memcpy(sync_block_out + i * layer1_size,
                     syn1neg + (size_t)sync_ind[i] * layer1_size,
                     layer1_size * sizeof(real));
            }

            MPI_Allreduce(MPI_IN_PLACE, sync_block_in,
                          sync_block_size * layer1_size, MPI_SCALAR, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, sync_block_out,
                          sync_block_size * layer1_size, MPI_SCALAR, MPI_SUM,
                          MPI_COMM_WORLD);

#pragma simd
#pragma vector aligneds
            for (size_t i = 0; i < sync_block_size * layer1_size; i++)
            {
              sync_block_in[i] /= num_procs;
              sync_block_out[i] /= num_procs;
            }

            for (size_t i = 0; i < sync_ind.size(); i++)
            {
              // printf("syn_ind: %d %d \n", i, sync_ind[i]);
              memcpy(syn0 + (size_t)sync_ind[i] * layer1_size,
                     sync_block_in + (i)*layer1_size,
                     layer1_size * sizeof(real));
              memcpy(syn1neg + (size_t)sync_ind[i] * layer1_size,
                     sync_block_out + (i)*layer1_size,
                     layer1_size * sizeof(real));
              // checkCUDAerr(cudaMemcpy(d_syn0 + (size_t)sync_ind[i] *
              // layer1_size,syn0 + (size_t)sync_ind[i] * layer1_size
              // ,layer1_size * sizeof(real) , cudaMemcpyHostToDevice));
              // checkCUDAerr(cudaMemcpy(d_syn1 + (size_t)sync_ind[i] *
              // layer1_size,syn1neg + (size_t)sync_ind[i] * layer1_size
              // ,layer1_size * sizeof(real) , cudaMemcpyHostToDevice));
            }
            Timer dvTimer;
            // sync_GPU_with_CPU(num_syncs);
            datamove_time += dvTimer.duration();
          }

          // let it go!
          compute_go = true;

          // print out status
          if (my_rank == 0 && debug_mode > 1)
          {
            double now = omp_get_wtime();
            printf("%cAlpha: %f  Progress : %.2f%%  Words/sec: %.2fk  Words "
                   "sync'ed: %d",
                   13, alpha, progress * 100,
                   word_count_actual_global / ((now - start) * 1000),
                   sync_vocab_size);
            fflush(stdout);
          }

          if (active_processes_global == 0)
          {
            break;
          }
          sync_start = omp_get_wtime();
        }
        else
        {
          usleep(1);
        }
      }
    }
    else
    {
      // printf("id %d  process compute\n",id);

      volatile size_t corpus_start = 0;
      volatile size_t corpus_end = 0;
      volatile size_t corpus_idx = corpus_start;

      long long word, word_count = 0, last_word_count = 0;
      long long local_iter = iter;
      // long long corpus_idx = 0;

      // use in kernel
      int total_sent_len, reduSize = 32;
      int *sen, *sentence_length, *d_sen, *d_sent_len;
      sen = (int *)malloc(MAX_SENTENCE * 20 * sizeof(int)); // ？？
      // checkCUDAerr(cudaHostAlloc((void **)&sen,MAX_SENTENCE * 20 *
      // sizeof(int) ,cudaHostAllocWriteCombined | cudaHostAllocMapped));
      sentence_length = (int *)malloc((MAX_SENTENCE + 1) * sizeof(int));
      // checkCUDAerr(cudaHostAlloc((void **)&sentence_length,(MAX_SENTENCE + 1)
      // * sizeof(int) ,cudaHostAllocWriteCombined | cudaHostAllocMapped));

      checkCUDAerr(
          cudaMalloc((void **)&d_sen, MAX_SENTENCE * 20 * sizeof(int)));
      //  checkCUDAerr(cudaHostGetDevicePointer(&d_sen, sen, 0));
      checkCUDAerr(
          cudaMalloc((void **)&d_sent_len, (1 + MAX_SENTENCE) * sizeof(int)));
      //  checkCUDAerr(cudaHostGetDevicePointer(&d_sent_len, sentence_length,
      //  0)); int *negSample;
      int *negSample = (int *)malloc(MAX_SENTENCE * negative * sizeof(int));
      // checkCUDAerr(cudaHostAlloc((void **)&negSample,MAX_SENTENCE * negative
      // * sizeof(int),cudaHostAllocWriteCombined | cudaHostAllocMapped));
      int *d_negSample;
      checkCUDAerr(
          cudaMalloc(&d_negSample, MAX_SENTENCE * negative * sizeof(int)));
      // checkCUDAerr(cudaHostGetDevicePointer(&d_negSample, negSample, 0));

      while (reduSize < layer1_size)
      {
        reduSize *= 2;
      }

      clock_t now;
      // FILE *fi = fopen(train_file, "rb");
      // fseek(fi, 0, SEEK_END);
      // uint64_t file_size = ftell(fi);
      // uint64_t  offset = (file_size / num_procs) * my_rank;
      // printf("[%d] offset: %ld file size: %ld\n", my_rank, offset,
      // file_size); fseek(fi, 0, SEEK_SET);

#pragma omp atomic
      ready_threads++;

      while (ready_threads < num_threads)
      {
        usleep(1);
      }

      // running until state is set to "OFF".
      int task_counter = 1;
      vector<int> corpus_segment;
      while (1)
      {
        if (trainer_state == TRAIN_OFF) // stop to train
        {
#pragma omp atomic
          active_threads--;
          break;
        }
        while (trainer_state == TRAIN_SUSPEND) // wait for notice
        {
          usleep(1);
          if (train_queue.empty())
            trainer_state = TRAIN_SUSPEND;
          else
          {
            queue_lock.lock();
            TrainTask &task = train_queue.front();
            corpus_segment = move(task.task_corpus);
            train_queue.pop();
            queue_lock.unlock();
            corpus_start = task.start_idx;
            corpus_end = task.end_idx;
            spdlog::debug("pull Train Task {0} {1}", corpus_start, corpus_end);
            cout << "====================== Task" << task_counter++ << " "
                 << task.start_idx << " " << task.end_idx
                 << " =======================" << endl;
            if (corpus_start >= corpus_end)
            {
              trainer_state = TRAIN_OFF;
              break;
            }
            trainer_state = TRAIN_ON;
          }
        }
        while (trainer_state == TRAIN_ON)
        {
          while (!compute_go)
          {
            usleep(1);
          }
          if (word_count - last_word_count > 10000)
          {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            // if ((debug_mode > 1)) {
            //   now = clock();
            //   printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13,
            //   alpha,
            //       word_count_actual / (float)(iter * train_words + 1) * 100,
            //       word_count_actual / ((float)(now - start + 1) /
            //       (float)CLOCKS_PER_SEC * 1000));
            //   fflush(stdout);
            // }
            alpha = starting_alpha *
                    (1 - word_count_actual / (float)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
              alpha = starting_alpha * 0.0001;
          }
          total_sent_len = 0;
          sentence_length[0] = 0;
          int cnt_sentence = 0;

          // load multiple sentence in one time
          while (cnt_sentence < MAX_SENTENCE)
          { // Read words
            int temp_sent_len = 0;
            char tSentence[MAX_SENTENCE_LENGTH];

            //   if (corpus_idx >= cu_local_corpus->size())
            //     break;
            if (corpus_idx >= corpus_end)
            {
              trainer_state = TRAIN_SUSPEND;
              spdlog::debug("Finish Train Task {0} {1}", corpus_start,
                            corpus_end);
              break;
            }

            // load a sentence in one circle
            while (1)
            {
              // if current corpus run out, trainer suspends.
              if (corpus_idx >= corpus_end)
              {
                trainer_state = TRAIN_SUSPEND;
                spdlog::debug("Finish load Train Task {0} {1}", corpus_start,
                              corpus_end);
                break;
              }
              // int origin_word = (*cu_local_corpus)[corpus_idx];
              int origin_word = corpus_segment[corpus_idx];
              corpus_idx++;
              if (origin_word == -1)
                break;
              if (origin_word < -1)
                exit(3);
              word_count++;
              word = hash2v_vocab[origin_word];
              // if (sample > 0)
              // { //  重采样
              //   float ran = (sqrt(v_vocab[word].cn / (sample * train_words))
              //   + 1) * (sample * train_words) / v_vocab[word].cn; int
              //   next_random_t = rand(); if (ran < (next_random_t & 0xFFFF) /
              //   (float)65536)
              //     continue;
              // }
              sen[total_sent_len] = word;
              total_sent_len++;
              temp_sent_len++;
              if (temp_sent_len >= MAX_SENTENCE_LENGTH)
                break;
            }

            cnt_sentence++;
            sentence_length[cnt_sentence] = total_sent_len;
            if (total_sent_len >= (MAX_SENTENCE - 1) * 20)
              break;

            //   char *wordTok;
            //   if (feof(fi))
            //     break;
            //   fgets(tSentence, MAX_SENTENCE_LENGTH + 1, fi); // 读取一个句子
            //   wordTok = strtok(tSentence, " \n\r\t");        //
            //   分解句子，获取每个单词 while (1)
            //   {
            //     if (wordTok == NULL)
            //     {
            //       word_count++;
            //       break;
            //     }
            //     word = SearchVocab(wordTok);
            //     wordTok = strtok(NULL, " \n\r\t");
            //     if (word == -1)
            //       continue;
            //     word_count++;
            //     if (word == 0)
            //     {
            //       word_count++;
            //       break;
            //     }
            //     if (sample > 0)
            //     { //  重采样
            //       float ran = (sqrt(vocab[word].cn / (sample * train_words))
            //       + 1) * (sample * train_words) / vocab[word].cn; int
            //       next_random_t = rand(); if (ran < (next_random_t & 0xFFFF)
            //       / (float)65536)
            //         continue;
            //     }
            //     sen[total_sent_len] = word;
            //     total_sent_len++;
            //     temp_sent_len++;
            //     if (temp_sent_len >= MAX_SENTENCE_LENGTH)
            //       break;
            //   }
            //   if (word == 0)
            //   {
            //     word_count++;
            //     break;
            //   }
            //   if (temp_sent_len >= MAX_SENTENCE_LENGTH)
            //     break;

            //   cnt_sentence++; // 句子的数量
            //   sentence_length[cnt_sentence] = total_sent_len;
            //   if (total_sent_len >= (MAX_SENTENCE - 1) * 20)
            //     break; // sen 装的一组句子的所有词
          }

          // if (feof(fi) || (word_count > train_words / num_procs))
          // TODO: consider multiple iterations.
          //         if (corpus_idx >= cu_local_corpus->size()) { // Initialize
          //         for iteration
          //           word_count_actual += word_count - last_word_count;
          //           local_iter--;
          //           if (local_iter == 0) {
          // #pragma omp atomic
          //             active_threads--;
          //             break;
          //           }
          //           word_count = 0;
          //           last_word_count = 0;
          //           for (int i = 0; i < MAX_SENTENCE + 1; i++)
          //             sentence_length[i] = 0;
          //           total_sent_len = 0;
          //           corpus_idx = 0;
          //           // fseek(fi, offset, SEEK_SET);
          //           continue;
          //         }

          // 先把 cnt_sentence 个句子的负样本采集好
          // Negative sampling in advance. A sentence shares negative samples
          for (int i = 0; i < cnt_sentence * negative; i++)
          {
            int randd = rand();
            int tempSample = table[randd % table_size];
            if (tempSample == 0)
              negSample[i] = randd % (vocab_size - 1) + 1;
            else
              negSample[i] = tempSample;
          }
          // printf("[ %d ] %d  %d\n",my_rank,cnt_sentence,total_sent_len);
          Timer dvTimer;
          checkCUDAerr(cudaMemcpy(d_negSample, negSample,
                                  cnt_sentence * negative * sizeof(int),
                                  cudaMemcpyHostToDevice));
          checkCUDAerr(cudaMemcpy(d_sen, sen, total_sent_len * sizeof(int),
                                  cudaMemcpyHostToDevice));
          checkCUDAerr(cudaMemcpy(d_sent_len, sentence_length,
                                  (cnt_sentence + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
          datamove_time += dvTimer.duration();
          if (cnt_sentence > MAX_SENTENCE)
            exit(1);

          // if (cbow)
          //   cbowKernel(d_sen, d_sent_len, alpha, cnt_sentence, reduSize);
          // else

          sgKernel(d_sen, d_sent_len, d_negSample, alpha, cnt_sentence,
                   reduSize);
          spdlog::debug("sgkernel work {0},{1}", corpus_start, corpus_end);

          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess)
          {
            printf("[ %d ]%s %d : %s\n", my_rank, __FILE__, __LINE__,
                   cudaGetErrorString(err));
            // Possibly: exit(-1) if program cannot continue....
          }
          dvTimer.restart();
          // checkCUDAerr(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size *
          // sizeof(float), cudaMemcpyDeviceToHost)); cudaMemcpy(syn1neg,
          // d_syn1, vocab_size * layer1_size * sizeof(float),
          // cudaMemcpyDeviceToHost));
          sync_CPU_with_GPU();
          // cudaDeviceSynchronize();

          datamove_time += dvTimer.duration();
        }
      }
      checkCUDAerr(cudaMemcpy(syn0, d_syn0,
                              vocab_size * layer1_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();

      // fclose(fi);

      // free memory
      free(sen);
      // free(sentence_length);
      free(negSample);
      // cudaFreeHost(sen);
      cudaFreeHost(sentence_length);
      // cudaFreeHost(negSample);
      cudaFree(d_sen);
      cudaFree(d_sent_len);
      cudaFree(d_negSample);
    }
  }
}

void TrainModel()
{
  long a, b, c, d;
  FILE *fo;
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0)
    ReadVocab();
  else
  {
    // LearnVocabFromTrainFile(); // 构建词表
    ConstVobcabFromMemory();
  }

  printf("[ %d ]Build Vocab ok\n", my_rank);

  if (save_vocab_file[0] != 0)
    SaveVocab();
  if (output_file[0] == 0)
    return;
  InitNet(); // 初始化 syn0 和 syn1 对应着 wih 和 woh ，并且在 GPU
             // 里申请空间，并且cpy过去
  printf("[ %d ]InitNet ok\n", my_rank);
  if (hs > 0)
    InitVocabStructCUDA();
  if (negative > 0)
    InitUnigramTable();
  printf("[ %d ]InitUnigramTable ok\n", my_rank);

  word_freq_block();

  start = clock();
  srand(time(NULL));

  printf("[ %d ]Preparatory work done\n", my_rank);
  Timer timer;

  TrainModelThread();

  printf("[ INFO ] Train Time : %f DataMove Time: %f\n", timer.duration(),
         datamove_time);

  // 释放空间
  cudaFree(d_table);
  cudaFree(d_syn1);
  cudaFree(d_syn0);
  cudaFree(d_vocab_codelen);
  cudaFree(d_vocab_point);
  cudaFree(d_vocab_code);

  // fo = fopen(output_file, "wb");
  // if (classes == 0)
  // {
  //   // Save the word vectors
  //   fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  //   for (a = 0; a < vocab_size; a++)
  //   {
  //     fprintf(fo, "%u ", v_vocab[a].id);
  //     if (binary)
  //       for (b = 0; b < layer1_size; b++)
  //         fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
  //     else
  //       for (b = 0; b < layer1_size; b++)
  //         fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
  //     fprintf(fo, "\n");
  //   }
  // }
  // else
  // {
  //   // Run K-means on the word vectors
  //   int clcn = classes, iter = 10, closeid;
  //   int *centcn = (int *)malloc(classes * sizeof(int));
  //   int *cl = (int *)calloc(vocab_size, sizeof(int));
  //   float closev, x;
  //   float *cent = (float *)calloc(classes * layer1_size, sizeof(float));

  //   for (a = 0; a < vocab_size; a++)
  //     cl[a] = a % clcn;
  //   for (a = 0; a < iter; a++)
  //   {
  //     for (b = 0; b < clcn * layer1_size; b++)
  //       cent[b] = 0;
  //     for (b = 0; b < clcn; b++)
  //       centcn[b] = 1;
  //     for (c = 0; c < vocab_size; c++)
  //     {
  //       for (d = 0; d < layer1_size; d++)
  //         cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
  //       centcn[cl[c]]++;
  //     }
  //     for (b = 0; b < clcn; b++)
  //     {
  //       closev = 0;
  //       for (c = 0; c < layer1_size; c++)
  //       {
  //         cent[layer1_size * b + c] /= centcn[b];
  //         closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
  //       }
  //       closev = sqrt(closev);
  //       for (c = 0; c < layer1_size; c++)
  //         cent[layer1_size * b + c] /= closev;
  //     }
  //     for (c = 0; c < vocab_size; c++)
  //     {
  //       closev = -10;
  //       closeid = 0;
  //       for (d = 0; d < clcn; d++)
  //       {
  //         x = 0;
  //         for (b = 0; b < layer1_size; b++)
  //           x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
  //         if (x > closev)
  //         {
  //           closev = x;
  //           closeid = d;
  //         }
  //       }
  //       cl[c] = closeid;
  //     }
  //   }

  // Save the K-means classes
  // for (a = 0; a < vocab_size; a++)
  //   fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);

  //   free(centcn);
  //   free(cent);
  //   free(cl);
  // }
  // fclose(fo);
}

int ArgPos(char *str, int argc, char **argv)
{
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a]))
    {
      if (a == argc - 1)
      {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int cuda_word2vec(int argc, char **argv, vector<int> *vertex_cn,
                  vector<int> *local_corpus)
{

  cu_vertex_cn = vertex_cn;
  cu_local_corpus = local_corpus;

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;

  // retrieve MPI task info
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Get_processor_name(hostname, &hostname_len);
  printf("[ %d ] vertex_cn : %ld local_corpus : %ld \n", my_rank,
         cu_vertex_cn->size(), cu_local_corpus->size());

  printf("processor name: %s, number of processors: %d, rank: %d \n", hostname,
         num_procs, my_rank);

  // for (int i=0 ; i< 100 ; i++) {
  //   printf("%d ",(*cu_local_corpus)[i]);
  // }

  int i;
  if (argc == 1)
  {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf(
        "\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with "
           "higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range "
           "is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 "
           "- 10 (0 = not used)\n");
    printf("\t-reuse-neg <int>\n");
    printf("\t\tA sentence share a negative sample set; (0 = not used / 1 = "
           "used)\n");

    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; "
           "default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram "
           "and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number "
           "of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf(
        "\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf(
        "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from "
           "the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for "
           "skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 "
           "-sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
    layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
    strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
    strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
    strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0)
    cbow = atoi(argv[i + 1]);
  if (cbow)
    alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
    alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
      window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
    sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0)
    hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
    iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0)
    classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-reuse-neg", argc, argv)) > 0)
    reuseNeg = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sync-size", argc, argv)) > 0)
    val_sync_vocab_size = atoi(argv[i + 1]);

  // 指定参数

  // strcpy(train_file,"/home/lzl/lzlmnt/GDistGER/corpus/wiki_corpus.txt");
  strcpy(output_file, "vectors.bin");

  vocab =
      (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int)); // 初始化为0
  expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));

  for (i = 0; i < EXP_TABLE_SIZE; i++)
  {
    expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP); // Precompute the exp() table
    expTable[i] =
        expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  checkCUDAerr(
      cudaMalloc((void **)&d_expTable,
                 (EXP_TABLE_SIZE + 1) * sizeof(float))); // 申请 device 空间
  checkCUDAerr(cudaMemcpy(
      d_expTable, expTable, (EXP_TABLE_SIZE + 1) * sizeof(float),
      cudaMemcpyHostToDevice)); // 把host的expTable 转移到 d_expTable 中

  printf("layer1_size: %lld sync_size: %d \n", layer1_size,
         val_sync_vocab_size);

  // cudaProfilerStart();
  TrainModel();
  // cudaProfilerStop();

  // memory free
  // free(vocab_codelen);
  // free(vocab_point);
  free(vocab_code);
  free(table);
  // free(syn0);
  free(syn1);
  // free(syn1neg);
  checkCUDAerr(cudaFreeHost(syn0));
  if (negative > 0)
    checkCUDAerr(cudaFreeHost(syn1neg));
  // free(vocab);
  // free(vocab_hash);
  free(expTable);
  cudaFree(d_expTable);

  return 0;
}
