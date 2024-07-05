#include "binding.h"

#include "lfu_cache.h"

using namespace cache;

PYBIND11_MODULE(lfucache, m) {
    py::class_<LFUCache>(m, "LFUCache")
        .def(py::init<size_t>())
        .def_property_readonly("limit", &LFUCache::getLimit)
        .def_property_readonly("perf", &LFUCache::getPerf)
        .def_property("perf_enabled", &LFUCache::getPerfEnabled,
                      &LFUCache::setPerfEnabled)
        .def("count", &LFUCache::count)
        .def("lookup", &LFUCache::lookup)
        .def("insert", &LFUCache::insert)
        .def("size", &LFUCache::size)
        .def("keys", &LFUCache::PyAPI_keys)
        .def("batchedLookup", &LFUCache::batchedLookup)
        .def("batchedInsert", &LFUCache::batchedInsert)
        .def("BatchedInsertArray", &LFUCache::BatchedInsertArray)
        .def("asyncBatchedInsertArray", &LFUCache::asyncBatchedInsertArray)
        .def("synchronize", &LFUCache::synchronize);

} // PYBIND11_MODULE
