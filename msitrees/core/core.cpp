// MIT License

// Copyright (c) 2020 xadrianzetx

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <limits>
#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor/xtensor.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xadapt.hpp>
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


std::tuple<int, xt::xarray<double>> class_proba(xt::pyarray<int>& y,
    size_t& n_classes) {
    // assuming that labels are encoded as 0 to n_classes, select
    // majority class in y and calculate array of probabilities
    // of selecting item from any given class
    if (y.dimension() != 1)
        throw py::value_error("Expected array with dim 1 in class_proba");

    if (y.shape(0) == 0)
        throw py::value_error("Empty array passed to class_proba");

    xt::xtensor<int, 1> counts = xt::bincount(y);
    int cls = *xt::argmax(counts).data();
    xt::xarray<double> proba = counts / (double)y.shape(0);

    if (proba.shape(0) < n_classes) {
        // classes with labels greater than max(y) still need
        // to be represented with probability of selecting item
        // from this class eq 0
        unsigned long long padlen = n_classes - proba.shape(0);
        proba = xt::pad(proba, {0, padlen}, xt::pad_mode::constant, 0.0);
    }

    std::tuple<int, xt::xarray<double>> params;
    params = {cls, proba};

    return params;
}

using quad = std::tuple<int, double, double, bool>;

quad cgbs(xt::pyarray<double>& x, xt::pyarray<int>& y,
    int& nfts, int& nobs) {
    // finds best tree split wrt. gini based information gain
    // to be used in classification tasks
    bool valid = false;
    int bestfeat = 0; 
    double bestsplt = 0.0;
    double importance = 0.0;
    double maxgain = -std::numeric_limits<double>::infinity();
    size_t ylen = y.shape(0);

    for (int i = 0; i < nfts; i++) {
        xt::xtensor<double, 1> col = (nfts != 1) ? xt::view(x, xt::all(), i) : x;
        xt::xtensor<double, 1> lvls = xt::unique(col);
        size_t nlvls = lvls.shape(0);
        
        for (int j = 0; j < nlvls; j++) {
            double lvl = lvls(j);
            xt::pyarray<int> left = xt::filter(y, col < lvl);
            xt::pyarray<int> right = xt::filter(y, col >= lvl);
            
            if (left.shape(0) == 0 || right.shape(0) == 0) {
                // split was proposed on either min or max
                // level, so all data is in either right or
                // left node - can skip this one
                continue;
            }
            
            double gain = gini_inf_gain(left, right, y);

            if (gain > maxgain) {
                valid = true;
                maxgain = gain;
                bestfeat = i;
                bestsplt = lvl;
                importance = (ylen / (double)nobs) * gain;
            }
        }
    }

    quad params {bestfeat, bestsplt, importance, valid};
    return params;
}

using idxpair = std::tuple<xt::xarray<int>, xt::xarray<int>>;

idxpair split_indices(xt::pyarray<double>& x, 
    std::tuple<int, double> params, int& nfts) {
    // split indices into left and right subtree
    // based on optimal split point
    int feat = std::get<0>(params);
    float split = std::get<1>(params);
    xt::xtensor<double, 1> col = (nfts != 1) ? xt::view(x, xt::all(), feat) : x;
    std::vector<size_t> idxl = *xt::where(col < split).data();
    std::vector<size_t> idxr = *xt::where(col >= split).data();
    idxpair indices {xt::adapt(idxl), xt::adapt(idxr)};

    return indices;
}


PYBIND11_MODULE(_core, m) {
    xt::import_numpy();
    m.def("num_classes", &num_classes);
    m.def("gini_impurity", &gini_impurity);
    m.def("entropy", &entropy);
    m.def("gini_information_gain", &gini_inf_gain);
    m.def("get_class_and_proba", &class_proba);
    m.def("classif_best_split", &cgbs);
    m.def("split_indices", &split_indices);
}
