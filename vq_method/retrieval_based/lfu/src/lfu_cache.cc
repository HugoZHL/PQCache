#include "lfu_cache.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace cache {

LFUCache::LFUCache(size_t limit) :
    limit_(limit), slot_cnt(0) {
}

int LFUCache::count(cache_key_t k) {
    return hash_.count(k);
}

void LFUCache::insert(cache_key_t e) {
    auto iter = hash_.find(e);
    if (iter == hash_.end()) {
        if (hash_.size() == limit_)
            _evict();
        hash_[e] = _create(e);
    } else {
        iter->second->ptr = e;
        hash_[e] = _increase(iter->second);
    }
}

int LFUCache::lookup(cache_key_t k) {
    auto iter = hash_.find(k);
    if (iter == hash_.end())
        return -1;
    hash_[k] = _increase(iter->second);
    auto result = iter->second->ptr;
    return result;
}

cache_key_t LFUCache::_evict() {
    auto clist = list_.begin();
    auto key = clist->list.back().ptr;
    hash_.erase(key);
    clist->list.pop_back();
    if (clist->list.empty())
        list_.erase(clist);
    return key;
}

std::list<LFUCache::Block>::iterator LFUCache::_create(int embed) {
    if (list_.empty() || list_.begin()->use > 1) {
        list_.push_front({std::list<Block>(), 1});
    }
    list_.begin()->list.push_front({embed, list_.begin()});
    return list_.begin()->list.begin();
}

std::list<LFUCache::Block>::iterator
LFUCache::_increase(std::list<Block>::iterator iter) {
    std::list<Block>::iterator result;
    auto clist = iter->head;
    auto clist_nxt = ++iter->head;
    size_t use = clist->use + 1;
    if (clist_nxt != list_.end() && clist_nxt->use == use) {
        clist_nxt->list.push_front({iter->ptr, clist_nxt});
        result = clist_nxt->list.begin();
    } else {
        CountList temp = {{}, use};
        auto clist_new = list_.emplace(clist_nxt, temp);
        clist_new->list.push_front({iter->ptr, clist_new});
        result = clist_new->list.begin();
    }
    clist->list.erase(iter);
    if (clist->list.empty())
        list_.erase(clist);
    return result;
}


vector<int> LFUCache::batchedLookup(const cache_key_t *keys,
                                             size_t len) {
    vector<int> result(len);
    for (size_t i = 0; i < len; i++) {
        int ptr = lookup(keys[i]);
        result[i] = ptr;
    }
    return result;
}

void LFUCache::batchedInsert(vector<cache_key_t> &ptrs) {
    for (auto &ptr : ptrs) {
        insert(ptr);
    }
}

void LFUCache::BatchedInsertArray(py::array_t<cache_key_t> &ptrs, py::array_t<int> &proxy) {
    py::buffer_info ptr_buf = ptrs.request();
    py::buffer_info proxy_buf = proxy.request();

    // Access the data pointers
    cache_key_t* ptrs_ = static_cast<cache_key_t*>(ptr_buf.ptr);
    int* proxy_ = static_cast<int*>(proxy_buf.ptr);

    for (size_t i = 0; i < ptr_buf.shape[0]; ++i) {
        cache_key_t e = ptrs_[i];
        auto iter = hash_.find(e);
        if (iter == hash_.end()) {
            int cur_slot;
            if (hash_.size() == limit_) {
                // std::cout<<"before evict"<<std::endl;
                auto evicted = _evict();
                // std::cout<<"after evict"<<std::endl;
                cur_slot = proxy_[evicted];
                proxy_[evicted] = -1;
            } else {
                cur_slot = (slot_cnt++);
            }
            hash_[e] = _create(e);
            proxy_[e] = cur_slot;
        } else {
            iter->second->ptr = e;
            hash_[e] = _increase(iter->second);
        }
    }
}

void LFUCache::asyncBatchedInsertArray(py::array_t<cache_key_t> &ptrs, py::array_t<int> &proxy) {
    task = std::async(std::launch::async, &LFUCache::BatchedInsertArray, this, std::ref(ptrs), std::ref(proxy));
}

void LFUCache::synchronize() {
    task.wait();
}

py::array_t<cache_key_t> LFUCache::PyAPI_keys() {
    std::vector<cache_key_t> keys;
    for (auto &iter : hash_) {
        keys.push_back(iter.first);
    }
    std::sort(keys.begin(), keys.end());
    return bind::vec(keys);
}

} // namespace cache
