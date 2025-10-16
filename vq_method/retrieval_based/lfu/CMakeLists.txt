cmake_minimum_required(VERSION 3.0)
aux_source_directory(src CACHE_SRC_LIST)

find_package(pybind11 2.6.0 CONFIG)

if (NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found")
else()
    pybind11_add_module(lfucache ${CACHE_SRC_LIST})
    target_include_directories(lfucache PUBLIC include)
endif()
