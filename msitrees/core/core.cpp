#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor-python/pyarray.hpp>

PYBIND11_MODULE(_core, m) {
    xt::import_numpy();
    
}
