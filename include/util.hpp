
#pragma once

#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include <random>
#include <chrono>

#include "type.hpp"
#include "constants.hpp"

class RandNumGenerator
{
public:
    virtual vertex_id_t gen(vertex_id_t upper_bound) = 0;
    virtual float gen_float(float upper_bound) = 0;
    virtual ~RandNumGenerator() {}
};

class StdRandNumGenerator : public RandNumGenerator
{
    std::random_device *rd;
    std::mt19937 *mt;
public:
    StdRandNumGenerator()
    {
        rd = new std::random_device();
        mt = new std::mt19937((*rd)());
    }
    ~StdRandNumGenerator()
    {
        delete mt;
        delete rd;
    }
    vertex_id_t gen(vertex_id_t upper_bound)
    {
        std::uniform_int_distribution<vertex_id_t> dis(0, upper_bound - 1);
        return dis(*mt);
    }
    float gen_float(float upper_bound)
    {
        std::uniform_real_distribution<float> dis(0.0, upper_bound);
        return dis(*mt);
    }
};

//Timer is used for performance profiling
class Timer
{
    std::chrono::time_point<std::chrono::system_clock> _start = std::chrono::system_clock::now();
public:
    void restart()
    {
        _start = std::chrono::system_clock::now();
    }
    double duration()
    {
        std::chrono::duration<double> diff = std::chrono::system_clock::now() - _start;
        return diff.count();
    }
    static double current_time()
    {
        std::chrono::duration<double> val = std::chrono::system_clock::now().time_since_epoch();
        return val.count();
    }
};

class MessageBuffer
{
    char padding[L1_CACHE_LINE_SIZE - sizeof(size_t) - sizeof(bool) - sizeof(vertex_id_t) - sizeof(void*)];
    bool is_private_data;
public:
    size_t sz;
    size_t count;
    void *data;
    MessageBuffer()
    {
        sz = 0;
        count = 0;
        data = nullptr;
        is_private_data = false;
    }
    MessageBuffer(size_t _sz, void* _data = nullptr) : MessageBuffer()
    {
#ifdef UNIT_TEST
        assert(data == nullptr);
#endif
        alloc(_sz, _data);
    }
    void alloc(size_t _sz, void *_data = nullptr)
    {
        if (data != nullptr && is_private_data == true)
        {
            delete [](char*)data;
        }
        sz = _sz;
        count = 0;
        if (_data == nullptr)
        {
            data = new char[sz];
            is_private_data = true;
        } else
        {
            data = _data;
            is_private_data = false;
        }
    }

    ~MessageBuffer()
    {
        if (data != nullptr && is_private_data == true)
        {
            delete [](char*)data;
        }
    }

    template<typename data_t>
    void write(data_t *val)
    {
        ((data_t*)data)[count++] = *val;
#ifdef UNIT_TEST
        self_check<data_t>();
#endif
    }

    void clear()
    {
        count = 0;
    }

    template<typename data_t>
    void self_check()
    {
        assert(sz >= count * sizeof(data_t));
    }
};
