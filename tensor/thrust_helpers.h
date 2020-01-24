#pragma once
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include "tensor_transprecision_cuda_procedures.h"
#include "broadcasting_helper.h"
#include "tensor.h"

namespace transprecision_floating_point
{
	struct broadcast_index_calculate
	{
		__host__ __device__ broadcast_index_calculate(size_t broadcast_size, size_t m, bool is_row) : broadcast_size_(broadcast_size), m_(m), is_row_(is_row) {  }

		__host__ __device__ size_t operator()(size_t index) const
		{
			return (is_row_ ? (index / m_) : (index % m_)) % broadcast_size_;
		}

		size_t broadcast_size_;
		size_t m_;
		bool is_row_;
	};

	template<typename ElementIterator>
	using broadcast_iterator = thrust::permutation_iterator<ElementIterator, thrust::transform_iterator<broadcast_index_calculate, thrust::counting_iterator<size_t>>>;

	template<typename UnaryFunction, typename T>
	struct unary_sanitize
	{
		__host__ __device__ unary_sanitize(
			UnaryFunction f,
			uint_fast8_t exp_bits,
			uint_fast8_t frac_bits) :
			f_(f), exp_bits_(exp_bits), frac_bits_(frac_bits)
		{ }

		__host__ __device__ T operator()(T value) const
		{
			return sanitize(exp_bits_, frac_bits_, f_(value));
		}

	private:
		UnaryFunction f_;
		uint_fast8_t exp_bits_;
		uint_fast8_t frac_bits_;
	};

	template<typename BinaryFunction, typename T, typename U, typename R>
	struct binary_sanitize
	{
		__host__ __device__ binary_sanitize(
			BinaryFunction f,
			uint_fast8_t exp_bits,
			uint_fast8_t frac_bits) :
			f_(f), exp_bits_(exp_bits), frac_bits_(frac_bits)
		{ }

		__host__ __device__ R operator()(T first, U second) const
		{
			return sanitize(exp_bits_, frac_bits_, f_(first, second));
		}

	private:
		BinaryFunction f_;
		uint_fast8_t exp_bits_;
		uint_fast8_t frac_bits_;
	};

	template<typename T>
	struct forward_element
	{
		__host__ __device__ forward_element(T element) : element_(element) {  }

		__host__ __device__ T operator()(T value) const
		{
			return element_;
		}

	private:
		T element_;
	};

	template<typename T, typename Function>
	struct forward_index
	{
		__host__ __device__ forward_index(T src, Function function, size_t size) : function_(std::move(function)), src_(std::move(src)), size_(size) {  }

		__host__ __device__ size_t operator()(size_t first, size_t second)
		{
			return function_(src_[first], src_[second]) ? first : second;
		}

	private:
		Function function_;
		T src_;
		size_t size_;
	};

	template <typename T>
	struct linear_index_to_row_index
	{
		__host__ __device__ linear_index_to_row_index(T c) : c_(c) {}

		__host__ __device__ T operator()(T i) const
		{
			return i / c_;
		}

		T c_;
	};

	template<typename BinaryFunction, typename T, typename R>
	struct axis_reducer
	{
		__host__ __device__ axis_reducer(
			size_t dim_size, size_t stride,
			T src, BinaryFunction function) :
			dim_size_(dim_size), stride_(stride),
			src_(src), function_(function)
		{  }

		__host__ __device__ R operator()(size_t index) const
		{
			auto id = (index % stride_) + stride_ * dim_size_ * (index / stride_);
			R r = src_[id];
			for (size_t i = 1; i < dim_size_; ++i)
				r = function_(r, src_[id + i * stride_]);

			return r;
		}

	private:
		size_t dim_size_;
		size_t stride_;
		T src_;
		BinaryFunction function_;
	};

	template<typename BinaryFunction, typename T, typename R>
	struct axis_reducer_index
	{
		__host__ __device__ axis_reducer_index(
			size_t dim_size, size_t stride,
			T src, BinaryFunction function) :
			dim_size_(dim_size), stride_(stride),
			src_(src), function_(function)
		{  }

		__host__ __device__ size_t operator()(size_t index) const
		{
			auto id = (index % stride_) + stride_ * dim_size_ * (index / stride_);
			size_t current_i = 0;
			for (size_t i = 1; i < dim_size_; ++i)
			{
				if (!function_(src_[id + current_i * stride_], src_[id + i * stride_]))
					current_i = i;
			}

			return current_i;
		}

	private:
		size_t dim_size_;
		size_t stride_;
		T src_;
		BinaryFunction function_;
	};

	struct linear_index_transpose
	{
		__host__ __device__ linear_index_transpose(size_t n, size_t m) : n_(n), m_(m) {  }

		__host__ __device__ size_t operator()(size_t index) const
		{
			auto const i = index / n_;
			auto const j = index % n_;
			return j * m_ + i;
		}

		size_t n_;
		size_t m_;
	};


	template<typename T, typename R>
	struct reducer
	{
		template <typename FirstSrc, typename FirstDst, typename BinaryFunction>
		static void reduce_by_axis(FirstSrc src, FirstDst dest, BinaryFunction function, tensor_shape const& shape,
			uint_fast8_t axis)
		{
			auto const dimensions = tensor_shape_dimension(shape);
			auto const total_size = tensor_shape_total_size(shape);

			auto dim_size = shape[axis];
			auto const new_size = total_size / dim_size;
			size_t upper_stride = total_size;
			for (size_t i = 0; i < axis; ++i)
				upper_stride /= shape[i];

			size_t stride = upper_stride / dim_size;

			if (axis == dimensions - 1)
			{
				thrust::device_vector<size_t> row_indices(new_size);
				thrust::reduce_by_key(
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_to_row_index<size_t>(dim_size)),
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_to_row_index<size_t>(dim_size)) + total_size,
					src,
					row_indices.begin(),
					dest,
					thrust::equal_to<size_t>(),
					function);
			}
			else
			{
				thrust::transform(
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
						axis_reducer<BinaryFunction, FirstSrc, R>(dim_size, stride, src, function)),
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
						axis_reducer<BinaryFunction, FirstSrc, R>(dim_size, stride, src, function)) +
					new_size,
					dest, thrust::identity<R>());
			}
		}

		template <typename FirstSrc, typename FirstDst, typename BinaryFunction>
		static void reduce_by_axis_index(FirstSrc src, FirstDst dest, BinaryFunction function, tensor_shape const& shape,
			uint_fast8_t axis)
		{
			auto const dimensions = tensor_shape_dimension(shape);
			auto const total_size = tensor_shape_total_size(shape);

			auto dim_size = shape[axis];
			auto const new_size = total_size / dim_size;
			size_t upper_stride = total_size;
			for (size_t i = 0; i < axis; ++i)
				upper_stride /= shape[i];

			size_t stride = upper_stride / dim_size;

			/*if (axis == dimensions - 1 && dim_size > 1)
			{
				thrust::device_vector<size_t> row_indices(new_size);
				thrust::reduce_by_key(
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_to_row_index<size_t>(dim_size)),
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_to_row_index<size_t>(dim_size)) + total_size,
					thrust::counting_iterator<size_t>(0),
					row_indices.begin(),
					dest,
					thrust::equal_to<size_t>(),
					forward_index<FirstSrc, decltype(function)>(src, function, dim_size));

				thrust::transform(dest, dest + new_size, dest, [dim_size]__device__(size_t index) { return index % dim_size; });
			}
			else*/
			{
				thrust::transform(
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
						axis_reducer_index<BinaryFunction, FirstSrc, R>(dim_size, stride, src, function)),
					thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
						axis_reducer_index<BinaryFunction, FirstSrc, R>(dim_size, stride, src, function)) +
					new_size,
					dest, thrust::identity<size_t>());
			}
		}
	};

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	template<typename FirstIterator, typename Broadcast, typename Destination, typename BinaryFunction>
	void broadcast_transform(FirstIterator first, Broadcast broadcast, Destination destination, BinaryFunction function,
		tensor_shape const& shape, tensor_shape const& broadcast_shape);

	template<typename ElementIterator>
	broadcast_iterator<ElementIterator> make_broadcast_iterator(ElementIterator first, tensor_shape const& shape, tensor_shape const& broadcast_shape);

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */

	template<typename FirstIterator, typename Broadcast, typename Destination, typename BinaryFunction>
	void broadcast_transform(FirstIterator first, Broadcast broadcast, Destination destination, BinaryFunction function,
		tensor_shape const& shape, tensor_shape const& broadcast_shape)
	{
		broadcasting_helper broadcasting(broadcast_shape, shape);
		if (!broadcasting.can_broadcast())
			throw std::runtime_error(
				"Invalid shapes - " + to_string(shape) + " and " + to_string(broadcast_shape) + "!");

		auto broadcast_size = broadcasting.tensor_size();
		auto total_size = tensor_shape_total_size(shape);

		if (broadcasting.axis() == 0)
		{
			auto m = total_size / shape.front();
			auto fun = [broadcast_size, m] __device__(size_t index)
			{
				return(index / m) % broadcast_size;
			};
			auto broadcast_first = thrust::make_permutation_iterator(broadcast, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), fun));
			thrust::transform(first, first + total_size,
				broadcast_first, destination,
				function);
		}
		else
		{
			auto m = total_size / shape.front();
			auto fun = [broadcast_size, m] __device__(size_t index)
			{
				return (index % m) % broadcast_size;
			};
			auto broadcast_first = thrust::make_permutation_iterator(broadcast, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), fun));
			thrust::transform(first, first + total_size,
				broadcast_first, destination,
				function);
		}
	}

	template <typename ElementIterator>
	broadcast_iterator<ElementIterator> make_broadcast_iterator(ElementIterator first, tensor_shape const& shape,
		tensor_shape const& broadcast_shape)
	{
		broadcasting_helper broadcasting(broadcast_shape, shape);
		if (!broadcasting.can_broadcast())
			throw std::runtime_error(
				"Invalid shapes - " + to_string(shape) + " and " + to_string(broadcast_shape) + "!");

		auto broadcast_size = broadcasting.tensor_size();
		auto total_size = tensor_shape_total_size(shape);

		if (broadcasting.axis() == 0)
		{
			auto m = total_size / shape.front();
			return thrust::make_permutation_iterator(first, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), broadcast_index_calculate(broadcast_size, m, true)));
		}

		auto m = total_size / shape.front();
		return thrust::make_permutation_iterator(first, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), broadcast_index_calculate(broadcast_size, m, false)));;
	}

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */
}