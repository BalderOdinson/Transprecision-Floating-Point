#pragma once
#include "tensor.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include "thrust_helpers.h"
#define FLOAT8 transprecision_floating_point::tensor_precision{ 5, 2 }
#define FLOAT16 transprecision_floating_point::tensor_precision{ 5, 10 }

namespace transprecision_floating_point
{
	inline tensor<half>::tensor() : precision_(FLOAT16)
	{
	}

	template <>
	template <typename U>
	tensor<half>::tensor(
		tensor<U> const& other) :
		data_(make_tensor_data<half>(tensor_shape_total_size(other.shape_))),
		shape_(other.shape()),
		precision_(other.precision_ > FLOAT16 ? FLOAT16 : other.precision_) 
	{
		thrust::device_ptr<half> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());
		auto total_size = tensor_shape_total_size(shape_);

		if (should_sanitize())
			thrust::transform(dev_ptr_other, dev_ptr_other + total_size, dev_ptr, unary_sanitize<thrust::identity<half>, half>(thrust::identity<half>(), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::copy(dev_ptr_other, dev_ptr_other + total_size, dev_ptr);
	}

	template <>
	inline tensor<half>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<half>(tensor_shape_total_size(shape))), shape_(shape), precision_(FLOAT16)
	{
	}

	inline bool tensor<half>::should_sanitize() const
	{
		return precision_ != FLOAT16;
	}

	template <>
	inline tensor<half>::tensor(tensor_shape const& shape, half default_element) :
		data_(make_tensor_data<half>(tensor_shape_total_size(shape))), shape_(shape), precision_(FLOAT16)
	{
		thrust::device_ptr<half> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<half>, half>(forward_element<half>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	inline void recursive_tensor_out_operator(std::ostream& os, tensor<half> const& obj, size_t dimension, tensor_index index)
	{
		if (dimension == tensor_shape_dimension(obj.shape()) - 1)
		{
			os << "[";
			index.push_back(0);
			size_t i = 0;
			for (; i < obj.shape()[dimension] - 1; ++i)
			{
				index.back() = i;
				os << float(obj[index]) << ", ";
			}
			index.back() = i;
			os << float(obj[index]);
			os << "]";

			return;
		}

		if (dimension == tensor_shape_dimension(obj.shape()) - 2)
		{
			index.push_back(0);
			for (size_t t = 0; t < dimension; ++t)
				os << "\t";
			os << "[";
			size_t i = 0;
			for (; i < obj.shape()[dimension] - 1; ++i)
			{
				index.back() = i;
				recursive_tensor_out_operator(os, obj, dimension + 1, index);
				os << ",\n";
				for (size_t t = 0; t < dimension; ++t)
					os << "\t";
			}
			index.back() = i;
			recursive_tensor_out_operator(os, obj, dimension + 1, index);
			os << "]";

			return;
		}

		index.push_back(0);
		for (auto i = 0; i < obj.shape()[dimension]; ++i)
		{
			index.back() = i;
			for (size_t t = 0; t < dimension; ++t)
				os << "\t";
			os << "[\n";
			recursive_tensor_out_operator(os, obj, dimension + 1, index);
			os << "\n";
			for (size_t t = 0; t < dimension; ++t)
				os << "\t";
			os << "]\n";
		}
	}

	inline std::ostream& operator<<(std::ostream& os, tensor<half> const& obj)
	{
		recursive_tensor_out_operator(os, obj, 0, {});
		return os;
	}
}