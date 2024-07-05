#pragma once

#include "binding.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <list>
#include <unordered_map>
#include <future>
#include <vector>
#include <queue>
using std::vector;

namespace cache {

typedef std::future<void> wait_t;
typedef int cache_key_t;

/*
  LFUCache:
    use LFU policy
    Implemented with hashmap and a 2-D list ordered by frequency
    O(1) insert and lookup
*/

class LFUCache {
private:
    size_t limit_;
    bool perf_enabled_ = false;
    py::list perf_;
    struct CountList;
    struct Block {
        int ptr;
        std::list<CountList>::iterator head;
    };
    struct CountList {
        std::list<Block> list;
        size_t use;
    };
    std::list<CountList> list_;
    std::unordered_map<cache_key_t, std::list<Block>::iterator> hash_;
    int slot_cnt;

    // helper function
    std::list<Block>::iterator _increase(std::list<Block>::iterator);
    std::list<Block>::iterator _create(int);
    cache_key_t _evict();

    wait_t task;

public:
    LFUCache(size_t limit);
    ~LFUCache() {
    }
    size_t getLimit() {
        return limit_;
    }
    size_t size() {
        return hash_.size();
    }
    int count(cache_key_t k);
    void insert(int e);
    int lookup(cache_key_t k);

    vector<int> batchedLookup(const cache_key_t *, size_t len);
    void batchedInsert(vector<cache_key_t> &ptrs);

    void BatchedInsertArray(py::array_t<cache_key_t> &ptrs, py::array_t<int> &proxy);
    void asyncBatchedInsertArray(py::array_t<cache_key_t> &ptrs, py::array_t<int> &proxy);
    void synchronize();

    //------------------------- implement main python API ---------------------
    bool getPerfEnabled() {
        return perf_enabled_;
    }
    void setPerfEnabled(bool value) {
        perf_enabled_ = value;
    }
    py::list getPerf() {
        return perf_;
    };
    // python debug function
    py::array_t<cache_key_t> PyAPI_keys();
}; // class LFUCache

} // namespace cache
