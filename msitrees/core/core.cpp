#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor-python/pyarray.hpp>


int num_classes(xt::pyarray<int>& y) {
    size_t cls = xt::unique(y).shape(0);
    return (int)cls;
}


xt::xtensor<int, 1> class_counts(xt::pyarray<int>& y, int& n_cls) {
    // initial implementation, change if too slow
    // returns counts of observations for each
    // unique discrete class
    xt::xtensor<int, 1> sorted = xt::sort(y);
    xt::xtensor<int, 1> cts = xt::zeros<int>({n_cls});

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


double gini_impurity(xt::pyarray<int>& y, int& n) {
    // Gini impurity is a measure of how often a randomly chosen
    // element from the set would be incorrectly labeled if it was 
    // randomly labeled according to the distribution of labels in the subset. 
    xt::xtensor<int, 1> counts = class_counts(y, n);
    xt::pyarray<float> probas = xt::pow(counts / (double)y.shape(0), 2);
    xt::xarray<double> gini = 1 - xt::sum(probas);
    
    return *gini.data();
}


PYBIND11_MODULE(_core, m) {
    xt::import_numpy();
    m.def("num_classes", num_classes);
    m.def("gini_impurity", gini_impurity);
}
