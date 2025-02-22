#pragma once
#include "core/common.h"
#include "core/data_type.h"
#include <iostream>
#include <random>

namespace infini {

class DataGenerator {
  private:
    virtual void fill(uint32_t *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); }

public:
    virtual ~DataGenerator() {}
    void operator()(void *data, size_t size, DataType dataType) {
        if (dataType == DataType::UInt32)
            fill(reinterpret_cast<uint32_t *>(data), size);
        else if (dataType == DataType::Float32)
            fill(reinterpret_cast<float *>(data), size);
        else
            IT_TODO_HALT();
    }
};

class IncrementalGenerator : public DataGenerator {
  public:
    virtual ~IncrementalGenerator() {}

  private:
    template <typename T> void fill(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};

template <int val> class ValGenerator : public DataGenerator {
  public:
    virtual ~ValGenerator() {}

  private:
    template <typename T> void fill(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = val;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};
typedef ValGenerator<1> OneGenerator;
typedef ValGenerator<0> ZeroGenerator;
} // namespace infini
