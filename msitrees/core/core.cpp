#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-python/pyarray.hpp>


int num_classes(xt::pyarray<int>& y) {
    size_t cls = xt::unique(y).shape(0);
    return (int)cls;
}


xt::xtensor<int, 1> class_counts(xt::pyarray<int>& y) {
    // initial implementation, change if too slow

    // sorted unique input gives N classes
    xt::xtensor<int, 1> sorted = xt::sort(y);
    xt::xtensor<int, 1> cls = xt::unique(sorted);
    xt::xtensor<int, 1> cts = xt::zeros<int>(cls.shape());

    int *idx = sorted.begin();
    int *end = sorted.end();
    int *cidx = cts.begin();
    int count = 0;

    while (idx != end) {
        count++;
        if (*idx != *(idx + 1)) {
            // next item belongs
            // to different class
            // so register current
            // class count and reset
            *cidx = count;
            count = 0;
            cidx++;
        }
        idx++;
    }

    return cts;
}


PYBIND11_MODULE(_core, m) {
    xt::import_numpy();
    m.def("num_classes", num_classes);
}
