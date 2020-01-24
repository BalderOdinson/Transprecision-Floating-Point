#pragma once
#include "tensor.h"
#include <curand.h>
#include "random_engine.h"
#include "tensor_math_operations_procedures.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include "thrust_helpers.h"
#include "tensor_half.h"

namespace transprecision_floating_point
{
	template <typename T>
	tensor<T>::tensor(
		tensor const& other) :
		data_(make_tensor_data<T>(tensor_shape_total_size(other.shape_))),
		shape_(other.shape_),
		precision_(other.precision_)
	{
		CHECK_CUDA_ERROR(cudaMemcpy(data_.get(), other.data_.get(), tensor_shape_total_size(shape_) * sizeof(T), cudaMemcpyHostToHost));
		update_values();
	}

	template <typename T>
	tensor<T>::tensor(
		tensor const& other,
		tensor_precision precision) :
		data_(make_tensor_data<T>(tensor_shape_total_size(other.shape_))),
		shape_(other.shape_),
		precision_(precision)
	{
		CHECK_CUDA_ERROR(cudaMemcpy(data_.get(), other.data_.get(), tensor_shape_total_size(shape_) * sizeof(T), cudaMemcpyHostToHost));
		update_values();
	}

	template <typename T>
	tensor<T>::tensor(
		tensor&& other) noexcept :
		data_(std::move(other.data_)),
		shape_(std::move(other.shape_)),
		precision_(other.precision_)
	{
	}

	template <typename T>
	tensor<T>::tensor(
		tensor&& other,
		tensor_precision precision) noexcept :
		data_(std::move(other.data_)),
		shape_(std::move(other.shape_)),
		precision_(precision)
	{
		if (other.precision_ != precision)
			update_values();
	}

	template <typename T>
	template <typename U>
	tensor<T>::tensor(
		tensor<U> const& other) :
		data_(make_tensor_data<T>(tensor_shape_total_size(other.shape_))),
		shape_(other.shape()),
		precision_(other.precision_)
	{
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());
		auto total_size = tensor_shape_total_size(shape_);

		if (should_sanitize())
			thrust::transform(dev_ptr_other, dev_ptr_other + total_size, dev_ptr, unary_sanitize<thrust::identity<T>, T>(thrust::identity<T>(), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::copy(dev_ptr_other, dev_ptr_other + total_size, dev_ptr);
	}

	template <typename T>
	template <typename U>
	tensor<T>::tensor(
		tensor<U> const& other,
		tensor_precision precision) :
		data_(make_tensor_data<T>(tensor_shape_total_size(other.shape_))),
		shape_(other.shape()),
		precision_(precision)
	{
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());
		auto total_size = tensor_shape_total_size(shape_);

		if (should_sanitize())
			thrust::transform(dev_ptr_other, dev_ptr_other + total_size, dev_ptr, unary_sanitize<thrust::identity<T>, T>(thrust::identity<T>(), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::copy(dev_ptr_other, dev_ptr_other + total_size, dev_ptr);
	}

	template <typename T>
	tensor<T>::tensor(tensor_shape const& shape, tensor_precision precision) :
		data_(make_tensor_data<T>(tensor_shape_total_size(shape))), shape_(shape), precision_(precision)
	{
	}

	template <typename T>
	tensor<T>::tensor(
		tensor_shape const& shape,
		tensor_precision precision,
		T default_element) :
		data_(make_tensor_data<T>(tensor_shape_total_size(shape))),
		shape_(shape),
		precision_(precision)
	{
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<T>, T>(forward_element<T>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <typename T>
	tensor<T>& tensor<T>::operator=(tensor const& other)
	{
		return (*this = tensor(other));
	}

	template <typename T>
	tensor<T>& tensor<T>::operator=(tensor&& other) noexcept
	{
		std::swap(shape_, other.shape_);
		std::swap(data_, other.data_);
		std::swap(precision_, other.precision_);

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator=(tensor<U> const& other)
	{
		return (*this = tensor(other));
	}

	template <typename T>
	void tensor<T>::set_precision(tensor_precision precision)
	{
		precision_ = precision;
		update_values();
	}

	template <typename T>
	tensor_precision tensor<T>::get_precision() const
	{
		return precision_;
	}

	template <typename T>
	T& tensor<T>::operator[](tensor_index const& idx)
	{
		if (tensor_shape_dimension(idx) != tensor_shape_dimension(shape_))
			throw std::runtime_error("Invalid index!");

		return data_[tensor_index_calculate(shape_, idx)];
	}

	template <typename T>
	T const& tensor<T>::operator[](tensor_index const& idx) const
	{
		if (tensor_shape_dimension(idx) != tensor_shape_dimension(shape_))
			throw std::runtime_error("Invalid index!");

		return data_[tensor_index_calculate(shape_, idx)];
	}

	template <typename T>
	tensor<T>& tensor<T>::operator+=(tensor const& other)
	{
		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::add(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator+=(tensor<U> const& other)
	{
		auto fun = [] __device__(T first, U second) { return arithmetic_operators<T, U, T>::add(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator+=(U other)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());

		auto fun = [other] __device__(T first) { return arithmetic_operators<T, U, T>::add(first, other); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::operator-=(tensor const& other)
	{
		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::sub(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator-=(tensor<U> const& other)
	{
		auto fun = [] __device__(T first, U second) { return arithmetic_operators<T, U, T>::sub(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator-=(U other)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());

		auto fun = [other] __device__(T first) { return arithmetic_operators<T, U, T>::sub(first, other); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::operator*=(tensor const& other)
	{
		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::mul(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator*=(tensor<U> const& other)
	{
		tensor cast_other = other;
		auto fun = [] __device__(T first, U second) { return arithmetic_operators<T, U, T>::mul(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_other(cast_other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator*=(U other)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());

		auto fun = [other] __device__(T first) { return arithmetic_operators<T, U, T>::mul(first, other); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::operator/=(tensor const& other)
	{
		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::div(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator/=(tensor<U> const& other)
	{
		auto fun = [] __device__(T first, U second) { return arithmetic_operators<T, U, T>::div(first, second); };
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<U> dev_ptr_other(other.data_.get());

		if (should_sanitize())
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits));
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, binary_sanitize<decltype(fun), T, U, T>(fun, precision_.exp_bits, precision_.frac_bits), shape_, other.shape_);
		}
		else
		{
			if (shape_ == other.shape_)
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr_other, dev_ptr, fun);
			else
				transprecision_floating_point::broadcast_transform(dev_ptr, dev_ptr_other, dev_ptr, fun, shape_, other.shape_);
		}

		return *this;
	}

	template <typename T>
	template <typename U>
	tensor<T>& tensor<T>::operator/=(U other)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());

		auto fun = [other] __device__(T first) { return arithmetic_operators<T, U, T>::div(first, other); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T> tensor<T>::operator~() const
	{
		tensor_shape new_shape(shape_.begin() + 1, shape_.end());
		new_shape.push_back(shape_.front());
		tensor result(new_shape, precision_);
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(result.data_.get());
		auto data_ptr = data_.get();

		auto n = shape_.front();
		auto m = total_size / n;

		auto transpose_fun = [data_ptr, n, m] __device__(size_t index)
		{
			auto const i = index / n;
			auto const j = index % n;
			return data_ptr[j * m + i];
		};

		thrust::transform(
			thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), transpose_fun),
			thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), transpose_fun) + total_size,
			dev_ptr, thrust::identity<T>());

		return result;
	}

	template <typename T>
	template <typename U>
	tensor<T> tensor<T>::dot(tensor<U> const& other) const
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

		tensor<float> a_cast(*this);
		tensor<float> b_cast(other);

		tensor<float> result(tensor_shape{ n,k });

		CHECK_CUBLAS_ERROR(cublasSgemm(tensor_lib_init::cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, k, n, m, &alpha, b_cast.data(), k, a_cast.data(), m, &beta, result.data(), k));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		return tensor(result, precision_);
	}

	template <typename T>
	tensor<T> tensor<T>::sum(uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor result(new_shape, precision_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_result(result.data_.get());

		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::add(first, second); };

		if (should_sanitize())
			reducer<T, T>::reduce_by_axis(
				dev_ptr, dev_ptr_result,
				binary_sanitize<decltype(fun), T, T, T>(
					fun, precision_.exp_bits, precision_.frac_bits),
				shape_, axis);
		else
			reducer<T, T>::reduce_by_axis(dev_ptr, dev_ptr_result, fun, shape_, axis);

		return result;
	}

	template <typename T>
	T tensor<T>::sum() const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first, T second) { return arithmetic_operators<T, T, T>::add(first, second); };
		if (should_sanitize())
			return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
		return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), fun);
	}

	template <typename T>
	tensor<T> tensor<T>::max(uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor result(new_shape, precision_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_result(result.data_.get());

		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::greater(first, second) ? first : second; };

		if (should_sanitize())
			reducer<T, T>::reduce_by_axis(
				dev_ptr, dev_ptr_result,
				binary_sanitize<decltype(fun), T, T, T>(
					fun, precision_.exp_bits, precision_.frac_bits),
				shape_, axis);
		else
			reducer<T, T>::reduce_by_axis(dev_ptr, dev_ptr_result, fun, shape_, axis);

		return result;
	}

	template <typename T>
	T tensor<T>::max() const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::greater(first, second) ? first : second; };
		if (should_sanitize())
			return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
		return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), fun);
	}

	template <typename T>
	tensor<T> tensor<T>::min(uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor result(new_shape, precision_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_result(result.data_.get());

		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::less(first, second) ? first : second; };

		if (should_sanitize())
			reducer<T, T>::reduce_by_axis(
				dev_ptr, dev_ptr_result,
				binary_sanitize<decltype(fun), T, T, T>(
					fun, precision_.exp_bits, precision_.frac_bits),
				shape_, axis);
		else
			reducer<T, T>::reduce_by_axis(dev_ptr, dev_ptr_result, fun, shape_, axis);

		return result;
	}

	template <typename T>
	T tensor<T>::min() const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::less(first, second) ? first : second; };
		if (should_sanitize())
			return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), binary_sanitize<decltype(fun), T, T, T>(fun, precision_.exp_bits, precision_.frac_bits));
		return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), fun);
	}

	template <typename T>
	tensor<size_t> tensor<T>::argmax(uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		auto const total_size = tensor_shape_total_size(shape_);

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor<size_t> result(new_shape);
		auto dim_size = shape_[axis];
		size_t upper_stride = total_size;
		for (size_t i = 0; i < axis; ++i)
			upper_stride /= shape_[i];

		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<size_t> dev_ptr_result(result.data_.get());

		auto fun = [] __host__ __device__(T first, T second) { return comparison_operators<T, T>::greater(first, second); };
		reducer<T, size_t>::reduce_by_axis_index(dev_ptr, dev_ptr_result, fun, shape_, axis);

		return result;
	}

	template <typename T>
	tensor_index tensor<T>::argmax() const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::less(first, second); };
		auto result = thrust::max_element(dev_ptr, dev_ptr + total_size, fun);
		return transprecision_floating_point::tensor_index_calculate(shape_, thrust::distance(dev_ptr, result));
	}

	template <typename T>
	tensor<size_t> tensor<T>::argmin(uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		auto const total_size = tensor_shape_total_size(shape_);

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor<size_t> result(new_shape);
		auto dim_size = shape_[axis];
		size_t upper_stride = total_size;
		for (size_t i = 0; i < axis; ++i)
			upper_stride /= shape_[i];
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<size_t> dev_ptr_result(result.data_.get());

		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::less(first, second); };
		reducer<T, size_t>::reduce_by_axis_index(dev_ptr, dev_ptr_result, fun, shape_, axis);

		return result;
	}

	template <typename T>
	tensor_index tensor<T>::argmin() const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first, T second) { return comparison_operators<T, T>::less(first, second); };
		auto result = thrust::min_element(dev_ptr, dev_ptr + total_size, fun);
		return transprecision_floating_point::tensor_index_calculate(shape_, thrust::distance(dev_ptr, result));
	}

	template <typename T>
	template <typename Reduction>
	tensor<T> tensor<T>::reduce(Reduction reduction, uint_fast8_t axis, bool keep_dims) const
	{
		auto const dimensions = tensor_shape_dimension(shape_);
		if (axis >= dimensions)
			throw std::runtime_error("Invalid axis!");

		tensor_shape new_shape = shape_;
		if (keep_dims)
			new_shape[axis] = 1;
		else
			new_shape.erase(new_shape.begin() + axis);
		tensor result(new_shape, precision_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		thrust::device_ptr<T> dev_ptr_result(result.data_.get());

		if (should_sanitize())
			reducer<T, T>::reduce_by_axis(
				dev_ptr, dev_ptr_result,
				binary_sanitize<Reduction, T, T, T>(
					reduction, precision_.exp_bits, precision_.frac_bits),
				shape_, axis);
		else
			reducer<T, T>::reduce_by_axis(dev_ptr, dev_ptr_result, reduction, shape_, axis);

		return result;
	}

	template <typename T>
	template <typename Reduction>
	T tensor<T>::reduce(Reduction reduction) const
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		if (should_sanitize())
			return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), binary_sanitize<Reduction, T, T, T>(reduction, precision_.exp_bits, precision_.frac_bits));
		return thrust::reduce(dev_ptr, dev_ptr + total_size, T(), reduction);
	}

	template <typename T>
	tensor<T>& tensor<T>::log()
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first) { return T(logf(float(first))); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::exp()
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first) { return T(expf(float(first))); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::abs()
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first) { return T(fabsf(float(first))); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::sqrt()
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [] __device__(T first) { return T(sqrtf(float(first))); };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::maximum(T max)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [max] __device__(T first) { return comparison_operators<T, T>::greater(first, max) ? first : max; };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	tensor<T>& tensor<T>::minimum(T min)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		auto fun = [min] __device__(T first) { return comparison_operators<T, T>::less(first, min) ? first : min; };
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fun), T>(fun, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, fun);

		return *this;
	}

	template <typename T>
	template <typename Function>
	tensor<T> tensor<T>::apply(Function func)
	{
		auto const total_size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<Function, T>(func, precision_.exp_bits, precision_.frac_bits));
		else
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, func);

		return *this;
	}

	template <typename T>
	void tensor<T>::reshape(tensor_shape const& shape)
	{
		if (tensor_shape_total_size(shape) != tensor_shape_total_size(shape_))
			throw std::runtime_error("Invalid shape!");

		shape_ = shape;
	}

	template <typename T>
	tensor_shape const& tensor<T>::shape() const
	{
		return shape_;
	}

	template <typename T>
	T* tensor<T>::data() const
	{
		return data_.get();
	}

	template <typename T>
	tensor<T> tensor<T>::create(tensor_shape const& shape, std::initializer_list<T> data)
	{
		auto const data_size = data.size();
		auto const total_size = tensor_shape_total_size(shape);

		if (data_size > 1 && data_size != total_size)
			throw std::runtime_error("Invalid data shape!");

		tensor t;
		t.shape_ = shape;
		t.data_ = make_tensor_data<T>(total_size);
		thrust::device_ptr<T> dev_ptr(t.data_.get());

		if (data_size == 1)
		{
			thrust::fill(dev_ptr, dev_ptr + total_size, *data.begin());
			return t;
		}

		thrust::device_vector<T> dev_data(data.begin(), data.end());
		thrust::copy(dev_data.begin(), dev_data.end(), dev_ptr);

		return t;
	}

	template <typename T>
	tensor<T> tensor<T>::create(tensor_shape const& shape, std::initializer_list<T> data,
		tensor_precision const& precision)
	{
		auto const data_size = data.size();
		auto const total_size = tensor_shape_total_size(shape);

		if (data_size > 1 && data_size != total_size)
			throw std::runtime_error("Invalid data shape!");

		tensor t;
		t.shape_ = shape;
		t.data_ = make_tensor_data<T>(total_size);
		t.precision_ = precision;

		thrust::device_ptr<T> dev_ptr(t.data_.get());

		if (data_size == 1)
		{
			auto v = data[0];
			auto fill_fun = [v] __device__ __host__(T value) { return v; };
			if (t.should_sanitize())
				thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<decltype(fill_fun), T>(fill_fun, t.precision_.exp_bits, t.precision_.frac_bits));
			else
				thrust::fill(dev_ptr, dev_ptr + total_size, data[0]);
			return t;
		}

		if (t.should_sanitize())
		{
			thrust::device_vector<T> dev_data(data);
			thrust::transform(dev_data.begin(), dev_data.end(), dev_ptr, unary_sanitize<thrust::identity<T>, T>(thrust::identity<T>(), t.precision_.exp_bits, t.precision_.frac_bits));
		}
		else
		{
			thrust::device_vector<T> dev_data(data);
			thrust::copy(dev_data.begin(), dev_data.end(), dev_ptr);
		}

		return t;
	}

	template <typename T>
	template <typename Distribution>
	tensor<T> tensor<T>::random(tensor_shape const& shape, Distribution distribution)
	{
		auto const total_size = tensor_shape_total_size(shape);
		tensor<float> t;
		t.shape_ = shape;
		t.data_ = make_tensor_data<float>(total_size);
		random_engine::generate(t.data_.get(), total_size, distribution);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		t.update_values();

		return tensor(std::move(t));
	}

	template <typename T>
	template <typename Distribution>
	tensor<T> tensor<T>::random(tensor_shape const& shape, Distribution distribution,
		tensor_precision const& precision)
	{
		auto const total_size = tensor_shape_total_size(shape);
		tensor<float> t;
		t.shape_ = shape;
		t.data_ = make_tensor_data<float>(total_size);
		t.precision_ = precision;
		random_engine::generate(t.data_.get(), total_size, distribution);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		t.update_values();

		return tensor(std::move(t));
	}

	template <typename T>
	void tensor<T>::update_values()
	{
		if (!should_sanitize())
			return;

		auto size = tensor_shape_total_size(shape_);
		thrust::device_ptr<T> dev_ptr(data_.get());

		thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, unary_sanitize<thrust::identity<T>, T>(thrust::identity<T>(), precision_.exp_bits, precision_.frac_bits));
	}

	template <typename T>
	bool tensor<T>::should_sanitize() const
	{
		return false;
	}

	template<typename P, typename U, typename R, typename Function>
	tensor<R> produce(tensor<P> const& lhs, tensor<U> const& rhs, Function function)
	{
		auto const total_size = tensor_shape_total_size(lhs.shape_);
		tensor<R> result(lhs.shape_);
		thrust::device_ptr<P> dev_ptr_lhs(lhs.data());
		thrust::device_ptr<U> dev_ptr_rhs(rhs.data());
		thrust::device_ptr<R> dev_ptr(result.data());
		if (result.should_sanitize())
			thrust::transform(dev_ptr_lhs, dev_ptr_lhs + total_size, dev_ptr_rhs, dev_ptr, binary_sanitize<Function, P, U, R>(function, result.precision_.exp_bits, result.precision_.frac_bits));
		else
			thrust::transform(dev_ptr_lhs, dev_ptr_lhs + total_size, dev_ptr_rhs, dev_ptr, function);
		return result;
	}

	template<typename P, typename U, typename R, typename Function>
	tensor<R> produce(tensor<P> const& lhs, tensor<U> const& rhs, Function function,
		tensor_precision const& precision)
	{
		auto const total_size = tensor_shape_total_size(lhs.shape_);
		tensor<R> result(lhs.shape_, precision);
		thrust::device_ptr<P> dev_ptr_lhs(lhs.data());
		thrust::device_ptr<U> dev_ptr_rhs(rhs.data());
		thrust::device_ptr<R> dev_ptr(result.data());
		if (result.should_sanitize())
			thrust::transform(dev_ptr_lhs, dev_ptr_lhs + total_size, dev_ptr_rhs, dev_ptr, binary_sanitize<Function, P, U, R>(function, result.precision_.exp_bits, result.precision_.frac_bits));
		else
			thrust::transform(dev_ptr_lhs, dev_ptr_lhs + total_size, dev_ptr_rhs, dev_ptr, function);
		return result;
	}

	template<typename T>
	void recursive_tensor_out_operator(std::ostream& os, tensor<T> const& obj, size_t dimension, tensor_index index)
	{
		if (dimension == tensor_shape_dimension(obj.shape()) - 1)
		{
			os << "[";
			index.push_back(0);
			size_t i = 0;
			for (; i < obj.shape()[dimension] - 1; ++i)
			{
				index.back() = i;
				os << obj[index] << ", ";
			}
			index.back() = i;
			os << obj[index];
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

	template<typename T>
	std::ostream& operator<<(std::ostream& os, tensor<T> const& obj)
	{
		recursive_tensor_out_operator(os, obj, 0, {});
		return os;
	}

	inline std::string to_string(tensor_shape const& shape)
	{
		std::ostringstream s;
		s << "(";
		for (size_t i = 0; i < tensor_shape_dimension(shape) - 1; ++i)
			s << shape[i] << ",";
		s << shape.back();
		s << ")";

		return s.str();
	}
}
