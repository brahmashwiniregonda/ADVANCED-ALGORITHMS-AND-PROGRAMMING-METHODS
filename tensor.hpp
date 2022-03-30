/**
 * @file tensor.hpp
 * 
 * @author Shehzad Inayat (888374)
 * @author Brahmashwini Regonda (887689)
 * 
 * @brief Tensor template class definition. 
 * @date 2021-12-01
 * 
 * 
 */
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <cassert>

/**
 * @brief Template container class for implementing a multi-dimensional array (or tensor)
 * with fixed size. 
 * 
 * @tparam T typename of the tensor value
 * @tparam Rank number of dimensions
 */
template <typename T, std::size_t Rank>
class tensor
{
protected:
    /**
     * @brief Dimensions array
     * 
     */
    std::array<std::size_t, Rank> dimensions;

    /**
     * @brief Storage array 
     * 
     */
    std::unique_ptr<T[]> storage;

    /**
     * @brief Computes the corresponding flat index for a given indices. 
     * Indices are computed using row major indexing.
     * Flat Index = ((0 x dim_0 + index_0) x dim_1 + index_1) x dim_2 + index_2 ...
     * 
     * @tparam Bounds checks if the function will check for bounds
     * @param indices array of indices to access
     * @return std::size_t flat index of the array of indices
     */
    template <bool Bounds = false>
    std::size_t storage_index(std::array<std::size_t, Rank> indices) const noexcept(!Bounds)
    {
        std::size_t flat_index = 0;

        // Loop through each dimension and compute the total flat index for the array
        for (std::size_t dim = 0; dim < Rank; dim++)
        {
            if constexpr (Bounds)
            {
                if (indices[dim] >= dimensions[dim])
                {
                    std::ostringstream msg;
                    msg << "tensor index #" << dim << " ("
                        << indices[dim] << ") is out of range";
                    throw std::out_of_range(msg.str());
                }
            }

            flat_index *= dimensions[dim];
            flat_index += indices[dim];
        }

        return flat_index;
    }

public:
    /**
     * @brief Reduces the array of indices to pin 
     * 
     * @tparam N number of pinned dimensions
     * @param index array of indices
     * @param pinned_dim array of dimensions with pins
     * @return std::array<std::size_t, Rank - N> reduced array of indices
     */
    template <std::size_t N>
    std::array<std::size_t, Rank - N> reduce_index(
        const std::array<std::size_t, Rank> &index,
        const std::array<bool, Rank> &pinned_dim) const
    {
        std::array<std::size_t, Rank - N> reduced_idx;

        auto reduced_idx_it = reduced_idx.begin();

        // Check all the data
        // Add to the reduced index array if unpinned
        for (std::size_t i = 0; i < Rank; i++)
        {
            if (!pinned_dim[i])
            {
                *reduced_idx_it = index[i];
                reduced_idx_it++;
            }
        }

        // Return the result
        return reduced_idx;
    }

    /**
     * @brief Increments the index array.
     * 
     * @param index array of indices to increment
     * @param pinned_dim array of pinned dimensions
     * @return true if the end was not reached
     * @return false otherwise
     */
    bool increment_index(std::array<std::size_t, Rank> &index,
                         const std::array<bool, Rank> &pinned_dim) const
    {
        // Find the first index that can be incremented
        // Start at the end of the index since it uses row-major indexing
        int i = Rank - 1;

        while (i >= 0 && pinned_dim[i])
            i--;

        // Check if all the dimension are pinned
        if (i < 0)
            return false;

        // Increment this index and carry to other dimensions
        while (i >= 0 && ++index[i] == dimensions[i])
        {
            // Set to zero
            index[i] = 0;

            // Move to next index
            i--;

            // Continue searching until the first unpinned dimension is found
            while (i >= 0 && pinned_dim[i])
                i--;
        }

        // Check if the end was not reached
        return i >= 0;
    }

    /**
     * @brief Construct a new tensor object using an array of dimension sizes.
     * 
     * @param dims array of dimension sizes
     */
    tensor(const std::array<std::size_t, Rank> &dims)
        : dimensions(dims),
          storage(std::make_unique<T[]>(
              std::accumulate(
                  dimensions.begin(), dimensions.end(), 1, std::multiplies<std::size_t>())))
    {
    }

    /**
     * @brief Construct a new tensor object using a variable argument dimensions.
     * 
     * @tparam Args variable size arguments of the dimension sizes.
     * @param args variable list of dimensions sizes.
     */
    template <typename... Args>
    tensor(Args... args)
        : tensor(std::array<std::size_t, Rank>{(std::size_t)args...})
    {
    }

    /**
     * @brief Access an item in the tensor object
     * 
     * @tparam Args variable list of indices
     * @param args variable list of indices
     * @return T& reference to the item
     */
    template <typename... Args>
    T &operator()(Args... args)
    {
        return (*this)({(std::size_t)args...});
    }

    /**
     * @brief Access an item in the tensor object
     * 
     * @param args array of indices
     * @return T& reference to the item
     */
    T &operator()(const std::array<std::size_t, Rank> &args)
    {
        return storage[storage_index<true>(args)];
    }

    /**
     * @brief Access an item in the constant tensor object
     * 
     * @tparam Args variable list of indices
     * @param args variable list of indices
     * @return const T& constant reference to the item
     */
    template <typename... Args>
    const T &operator()(Args... args) const
    {
        return (*this)({(std::size_t)args...});
    }

    /**
     * @brief Access an item in the tensor object
     * 
     * @param args array of indices
     * @return T& reference to the item
     */
    const T &operator()(const std::array<std::size_t, Rank> &args) const
    {
        return storage[storage_index<true>(args)];
    }

    // Slice
    /**
     * @brief Slice the tensor object given an array of (dimension, index). The list 
     * of (dimension, index) are used to as reference for the list of items to retain.
     * 
     * @tparam N length of the args.
     * @param args array of (dimension, index) tuples
     * @return tensor<T, Rank - N / 2> reduced tensor object
     */
    template <std::size_t N>
    tensor<T, Rank - N / 2> slice(const std::array<std::size_t, N> &args)
    {
        // Check if N is divisible by 2
        if (N % 2 != 0)
        {
            throw std::invalid_argument("Missing pair in the arguments");
        }

        // Compute the number of dimensions to pin
        auto n = N / 2;

        // Check Rank >= n
        if (Rank < n)
        {
            throw std::invalid_argument("Number of dimensions to slice exceeds tensor rank");
        }

        // Split the args into pinned dimensions and pinned indices
        std::array<bool, Rank> pinned_dim{false};
        std::array<std::size_t, Rank> pinned_idx{0};

        // Copy the args to appropriate array
        for (std::size_t i = 0; i < n; i++)
        {
            // Dimensions should be pinned only once
            if (pinned_dim[args[i * 2]])
            {
                throw std::invalid_argument("Dimension is pinned twice");
            }

            pinned_dim[args[i * 2]] = 1;
            pinned_idx[args[i * 2]] = args[i * 2 + 1];
        }

        // Create a new dimension array from the pinned dimensions
        std::array<std::size_t, Rank - N / 2> new_dim;
        auto new_dim_it = new_dim.begin();

        for (std::size_t i = 0; i < Rank; i++)
        {
            // Check if dimension is pinned
            // Copy dimension if it is not pinned
            if (!pinned_dim[i])
            {
                *new_dim_it = dimensions[i];
                new_dim_it++;
            }
        }

        // Create the sliced tensor
        auto sliced_t = tensor<T, Rank - N / 2>(new_dim);

        // Fill up the sliced tensor with data
        // Loop through all the possible values of pinned_idx and copy it to the new
        do
        {
            // Reduce the dimension and copy the value from the tensor
            sliced_t(reduce_index<N / 2>(pinned_idx, pinned_dim)) = (*this)(pinned_idx);

        } while (increment_index(pinned_idx, pinned_dim));

        return sliced_t;
    }

    /**
     * @brief Slice the tensor object given an array of (dimension, index). The list 
     * of (dimension, index) are used to as reference for the list of items to retain.
     * 
     * @tparam variable list of arguments
     * @param args list of (dimension, index) tuples
     * @return tensor<T, Rank - N(args) / 2> reduced tensor object
     */
    template <typename... Args>
    tensor<T, Rank - sizeof...(Args) / 2> slice(Args... args)
    {
        return slice(std::array<std::size_t, sizeof...(Args)>{(std::size_t)args...});
    }

    /**
     * @brief Gets the array of dimension sizes of the tensor.
     * 
     * @return const std::array<std::size_t, Rank> array of dimensions.
     */
    const std::array<std::size_t, Rank> &size() const
    {
        return dimensions;
    }

    /**
     * @brief Computes the tensor product. If c is the result, then the tensor product
     * is given as c_{i,j,k} = \sum_{n=0}^1\sum_{m=0}^2 a_{n,m,i} * b_{j,m,k,n}.
     * 
     * @tparam OtherRank Rank of the other tensor
     * @tparam N number of indices
     * @param left tensor 1
     * @param right tensor 2
     * @param args pairs of indices to contract (summed over)
     * @return tensor<T, Rank + OtherRank - N> product
     */
    template <std::size_t OtherRank, std::size_t N>
    friend tensor<T, Rank + OtherRank - N> tensor_product(
        const tensor<T, Rank> &left,
        const tensor<T, OtherRank> &right,
        const std::array<std::size_t, N> &args)
    {
        // Check if N exceeds the sum of ranks
        if (Rank + OtherRank < N)
        {
            throw std::invalid_argument("Number of contracted indices exceeds total dimensions");
        }

        // Check if N is divisible by two
        if (N % 2 != 0)
        {
            throw std::invalid_argument("A dimension has no pair");
        }

        // Compute the number of pairs of indices
        auto n = N / 2;

        // Create the pinned indices array
        std::array<bool, Rank> left_pinned_dims{false};
        std::array<bool, OtherRank> right_pinned_dims{false};

        // Fill up the pinned indices
        for (std::size_t i = 0; i < n; i++)
        {
            auto left_pinned_idx = args[i * 2];
            auto right_pinned_idx = args[i * 2 + 1];

            // Check if the left and right pinned indices have the same size
            if (left.dimensions[left_pinned_idx] != right.size()[right_pinned_idx])
            {
                throw std::invalid_argument("Contracted indices must have the same size");
            }

            left_pinned_dims[left_pinned_idx] = true;
            right_pinned_dims[right_pinned_idx] = true;
        }

        // Get the unpinned indices
        std::array<bool, Rank> left_unpinned_dims{false};
        std::array<bool, OtherRank> right_unpinned_dims{false};

        // Fill the unpinned indices array
        std::transform(left_pinned_dims.begin(), left_pinned_dims.end(), left_unpinned_dims.begin(), std::logical_not<bool>());
        std::transform(right_pinned_dims.begin(), right_pinned_dims.end(), right_unpinned_dims.begin(), std::logical_not<bool>());

        // Initialize the index arrays for the computation
        std::array<std::size_t, Rank> left_index{0};
        std::array<std::size_t, OtherRank> right_index{0};
        std::array<std::size_t, Rank - (N / 2)> reduced_left_index;
        std::array<std::size_t, OtherRank - (N / 2)> reduced_right_index;
        std::array<std::size_t, Rank + OtherRank - N> combined_index;

        // Compute the dimension array for the result
        reduced_left_index = left.reduce_index<N / 2>(left.dimensions, left_pinned_dims);
        reduced_right_index = right.template reduce_index<N / 2>(right.size(), right_pinned_dims);

        // Concatenate the reduced index arrays
        std::copy(reduced_left_index.begin(), reduced_left_index.end(), combined_index.begin());
        std::copy(reduced_right_index.begin(), reduced_right_index.end(), combined_index.begin() + Rank - n);

        // Initialize the results tensor
        tensor<T, Rank + OtherRank - N> result(combined_index);

        // Reset the indices
        reduced_left_index = {0};
        reduced_right_index = {0};

        // Outer loop is to fill up each item in the result tensor
        // Start looping
        do
        {
            // Reduce the indices
            reduced_left_index = left.reduce_index<N / 2>(left_index, left_pinned_dims);
            reduced_right_index = right.template reduce_index<N / 2>(right_index, right_pinned_dims);

            // Concatenate the reduced index arrays
            std::copy(reduced_left_index.begin(), reduced_left_index.end(), combined_index.begin());
            std::copy(reduced_right_index.begin(), reduced_right_index.end(), combined_index.begin() + Rank - n);

            // Initialize items to zero
            result(combined_index) = 0;

            // Perform the summation
            bool cont = false;
            do
            {
                // Compute the product between two components
                result(combined_index) += left(left_index) * right(right_index);

                // Increment the left and right indices to continue summing the contracted indices
                cont = left.increment_index(left_index, left_unpinned_dims);
                cont &= right.increment_index(right_index, right_unpinned_dims);
            } while (cont);

        } while (left.increment_index(left_index, left_pinned_dims) || right.increment_index(right_index, right_pinned_dims));

        return result;
    }

    /**
     * @brief Computes the tensor product. If c is the result, then the tensor product
     * is given as c_{i,j,k} = \sum_{n=0}^1\sum_{m=0}^2 a_{n,m,i} * b_{j,m,k,n}.
     * 
     * @tparam OtherRank Rank of tensor 2
     * @tparam Args list of indices
     * @param left tensor 1
     * @param right tensor 2
     * @param args number of indices to contract summed over
     * @return tensor<T, Rank + OtherRank - sizeof...(Args)> product
     */
    template <std::size_t OtherRank, typename... Args>
    friend tensor<T, Rank + OtherRank - sizeof...(Args)> tensor_product(
        const tensor<T, Rank> &left, const tensor<T, OtherRank> &right, Args... args)
    {
        return tensor_product(left, right, std::array<std::size_t, sizeof...(Args)>{(std::size_t)args...});
    }
};

#endif
