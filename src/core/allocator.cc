#include "core/allocator.h"
#include "core/common.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t offset = 0;
        if (freeBlock.size() == 0) {
            offset = peak;
            peak += size;
        } else {
            auto it = freeBlock.begin();
            for (; it != freeBlock.end(); it++) {
                if (it->second >= size) {
                    break;
                }
            }
            
            if (it != freeBlock.end()) {
                offset = it->first;
                auto remain = it->second - size;
                freeBlock.erase(it);
                if (remain >= 0) {
                    freeBlock[offset + size] = remain;
                }
            } else {
                it = std::prev(it);
                if (it->first + it->second == peak) {
                    offset = it->first;
                    peak += size - it->second;
                    freeBlock.erase(it);
                } else {
                    offset = peak;
                    peak += size;
                }
            }
        }
        used += size;

        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t begin;
        size_t blockSize = 0;

        auto left = std::find_if(freeBlock.begin(), freeBlock.end(), [&addr](const std::pair<size_t, size_t> &item) {return item.first + item.second == addr;});
        if (left != freeBlock.end()) {
            begin = left->first;
            blockSize += left->second;
            freeBlock.erase(left);
        } else {
            begin = addr;
        }
        blockSize += size;
        auto right = std::find_if(freeBlock.begin(), freeBlock.end(), [&addr, &size] (const std::pair<size_t, size_t> &item) {return item.first == addr + size;});
        if (right != freeBlock.end()) {
            blockSize += right->second;
            freeBlock.erase(right);
        }

        used -= size;
        freeBlock[begin] = blockSize;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
