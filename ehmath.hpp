#ifndef EHMATH_HPP
#define EHMATH_HPP

#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdlib.h>


template <typename T>
void median1d(std::vector<T> & out, std::vector<T> const& in, int msize);

template <typename T>
T median_val_filter(std::vector<T> & in);

/**
 *  Standard median 1d filter.
 *  Doesn't perform in-place filtering, so
 *  arrays should be pointers to physically different memory places.
 *
 * \param[out] - output filtered array(time series), memory must be preallocated, length must match with length of 'in';
 * \param[in] - input array(time series);
 * \param[in] - size of sliding mask, odd number = {3, 5, 7, ..., l / 2 - 1 }.
 *
 */
template <typename T>
void
median1d(std::vector<T> & out, std::vector<T> const& in, int msize) {
    int l = in.size();
    // Run in array over with sliding mask.
    int shift = msize / 2;
    for(int i = shift; i < l - shift; ++i) {
        // Get current mask.
        std::vector<T> mask(msize);
        for(int j = -shift; j < shift; ++j) {
            mask[shift + j] = in[i + j];
        }

        // Sort mask in descending or ascending order.
        std::sort(mask.begin(), mask.end());

        // Get pivoting element.
        float pivot = mask[shift];

        // Set it to output array.
        out[i] = pivot;
    }
}


template <typename T>
T median_val_filter(std::vector<T> & in) {
    if(in.empty())
        return 0;

    if(in.size() <= 2)
        return in.front();

    std::sort(in.begin(), in.end());

    size_t shift = size_t(in.size() / 2);

    return in[shift];
}


#endif // EHMATH_HPP
