#pragma once
#include "tensor.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include "thrust_helpers.h"
#define FLOAT16ALT transprecision_floating_point::tensor_precision{ 8, 7 }
#define FLOAT32 transprecision_floating_point::tensor_precision{ 8, 23 }


namespace transprecision_floating_point
{
	template <>
	inline tensor<float>::tensor() : precision_(FLOAT32)
	{
	}

	template <>
	inline tensor<float>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<float>(tensor_shape_total_size(shape))), shape_(shape), precision_(FLOAT32)
	{
	}

	template <>
	inline bool tensor<float>::should_sanitize() const
	{
		return precision_ != FLOAT32;
	}

	template <>
	inline tensor<float>::tensor(tensor_shape const& shape, float default_element) :
		data_(make_tensor_data<float>(tensor_shape_total_size(shape))), shape_(shape), precision_(FLOAT32)
	{
		thrust::device_ptr<float> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<float>, float>(forward_element<float>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<float> tensor<float>::operator~() const
	{
		tensor_shape new_shape(shape_.begin() + 1, shape_.end());
		new_shape.push_back(shape_.front());
		tensor result(new_shape, precision_);
		auto const total_size = tensor_shape_total_size(shape_);

		float const alpha(1.0);
		float const beta(0.0);
		auto n = shape_.front();
		auto m = total_size / n;

		CHECK_CUBLAS_ERROR(cublasSgeam(tensor_lib_init::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, data_.get(), m, &beta, data_.get(), n, result.data_.get(), n));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		return result;
	}

	template <>
	template <typename U>
	tensor<float> tensor<float>::dot(tensor<U> const& other) const
	{
		auto const total_size_first = tensor_shape_total_size(shape_);
		auto const total_size_second = tensor_shape_total_size(other.shape_);
		auto n = shape_.front();
		auto m = other.shape_.front();
		auto k = total_size_second / m;

		if (m != total_size_first / n)
			throw std::runtime_error("Invalid shapes - " + to_string(shape_) + " and " + to_string(other.shape_) + "!");

		float const alpha(1.0);
		float const beta(0.0);

		tensor<float> b_cast(other);

		tensor<float> result(tensor_shape{ n,k }, precision_);

		CHECK_CUBLAS_ERROR(cublasSgemm(tensor_lib_init::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, n, m, &alpha, b_cast.data(), k, data_.get(), m, &beta, result.data(), k));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		result.update_values();

		return result;
	}

	template <>
	template <>
	inline tensor<float> tensor<float>::dot(tensor<float> const& other) const
	{
		auto const total_size_first = tensor_shape_total_size(shape_);
		auto const total_size_second = tensor_shape_total_size(other.shape_);
		auto n = shape_.front();
		auto m = other.shape_.front();
		auto k = total_size_second / m;

		if (m != total_size_first / n)
			throw std::runtime_error("Invalid shapes - " + to_string(shape_) + " and " + to_string(other.shape_) + "!");

		float const alpha(1.0);
		float const beta(0.0);

		tensor<float> result(tensor_shape{ n,k }, precision_);

		CHECK_CUBLAS_ERROR(cublasSgemm(tensor_lib_init::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, n, m, &alpha, other.data(), k, data_.get(), m, &beta, result.data(), k));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		result.update_values();

		return result;
	}
}
