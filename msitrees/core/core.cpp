#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xhistogram.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor-python/pyarray.hpp>


namespace py = pybind11;


int num_classes(xt::pyarray<int>& y) {
    size_t cls = xt::unique(y).shape(0);
    return (int)cls;
}


double gini_impurity(xt::pyarray<int>& y) {
    // Gini impurity is a measure of how often a randomly chosen
    // element from the set would be incorrectly labeled if it was 
    // randomly labeled according to the distribution of labels in the subset.

    if (y.dimension() != 1)
        throw py::value_error("Expected array with dim 1 in gini_impurity");

    if (y.shape(0) == 0)
        throw py::value_error("Empty array passed to gini_impurity");

    xt::xtensor<int, 1> counts = xt::bincount(y);
    xt::xtensor<double, 1> probas = xt::pow(counts / (double)y.shape(0), 2);
    xt::xarray<double> gini = 1 - xt::sum(probas);
    
    return *gini.data();
}


double entropy(xt::pyarray<int>& y) {
    // https://en.wikipedia.org/wiki/Entropy_(information_theory)

    if (y.dimension() != 1)
        throw py::value_error("Expected array with dim 1 in entropy");

    if (y.shape(0) == 0)
        throw py::value_error("Empty array passed to entropy");

    xt::xtensor<int, 1> counts = xt::bincount(y);
    // fix bincount output for non-consecutive class labels
    // or multiclass not including 0
    xt::xtensor<int, 1> fcounts = xt::filter(counts, counts > 0);
    xt::xtensor<double, 1> probas = fcounts / (double)y.shape(0);
    xt::xarray<double> entropy = xt::sum(-probas * xt::log2(probas));

    return *entropy.data();
}


double gini_inf_gain(xt::pyarray<int>& left,
    xt::pyarray<int>& right, xt::pyarray<int>& all) {
    // https://en.wikipedia.org/wiki/Information_gain_in_decision_trees

    double gini_left = gini_impurity(left);
    double gini_right = gini_impurity(right); 
    double h = gini_impurity(all);

    double dimall = (double)all.shape(0);
    double hl = ((double)left.shape(0) / dimall) * gini_left;
    double hr = ((double)right.shape(0) / dimall) * gini_right;
    double gain = h - (hl + hr);

    return gain;
}


PYBIND11_MODULE(_core, m) {
    xt::import_numpy();
    m.def("num_classes", &num_classes);
    m.def("gini_impurity", &gini_impurity);
    m.def("entropy", &entropy);
    m.def("gini_information_gain", &gini_inf_gain);
}
