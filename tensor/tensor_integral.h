#pragma once
#include "tensor.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include "thrust_helpers.h"
#define INT8 transprecision_floating_point::tensor_precision{ 0, 8 }
#define UINT8 transprecision_floating_point::tensor_precision{ 0, 8 }
#define INT16 transprecision_floating_point::tensor_precision{ 0, 16 }
#define UINT16 transprecision_floating_point::tensor_precision{ 0, 16 }
#define INT32 transprecision_floating_point::tensor_precision{ 0, 32 }
#define UINT32 transprecision_floating_point::tensor_precision{ 0, 32 }
#define INT64 transprecision_floating_point::tensor_precision{ 0, 64 }
#define UINT64 transprecision_floating_point::tensor_precision{ 0, 64 }

namespace transprecision_floating_point
{
	template <>
	inline tensor<int64_t>::tensor() : precision_(INT64)
	{
	}

	template <>
	inline tensor<int64_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<int64_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT64)
	{
	}

	template <>
	inline tensor<int64_t>::tensor(tensor_shape const& shape, int64_t default_element) :
		data_(make_tensor_data<int64_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT64)
	{
		thrust::device_ptr<int64_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<int64_t>, int64_t>(forward_element<int64_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<uint64_t>::tensor() : precision_(UINT64)
	{
	}

	template <>
	inline tensor<uint64_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<uint64_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT64)
	{
	}

	template <>
	inline tensor<uint64_t>::tensor(tensor_shape const& shape, uint64_t default_element) :
		data_(make_tensor_data<uint64_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT64)
	{
		thrust::device_ptr<uint64_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<uint64_t>, uint64_t>(forward_element<uint64_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<int32_t>::tensor() : precision_(INT32)
	{
	}

	template <>
	inline tensor<int32_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<int32_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT32)
	{
	}

	template <>
	inline tensor<int32_t>::tensor(tensor_shape const& shape, int32_t default_element) :
		data_(make_tensor_data<int32_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT32)
	{
		thrust::device_ptr<int32_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<int32_t>, int32_t>(forward_element<int32_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<uint32_t>::tensor() : precision_(UINT32)
	{
	}

	template <>
	inline tensor<uint32_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<uint32_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT32)
	{
	}

	template <>
	inline tensor<uint32_t>::tensor(tensor_shape const& shape, uint32_t default_element) :
		data_(make_tensor_data<uint32_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT32)
	{
		thrust::device_ptr<uint32_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<uint32_t>, uint32_t>(forward_element<uint32_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<int16_t>::tensor() : precision_(INT16)
	{
	}

	template <>
	inline tensor<int16_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<int16_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT16)
	{
	}

	template <>
	inline tensor<int16_t>::tensor(tensor_shape const& shape, int16_t default_element) :
		data_(make_tensor_data<int16_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT16)
	{
		thrust::device_ptr<int16_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<int16_t>, int16_t>(forward_element<int16_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<uint16_t>::tensor() : precision_(UINT16)
	{
	}

	template <>
	inline tensor<uint16_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<uint16_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT16)
	{
	}

	template <>
	inline tensor<uint16_t>::tensor(tensor_shape const& shape, uint16_t default_element) :
		data_(make_tensor_data<uint16_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT16)
	{
		thrust::device_ptr<uint16_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<uint16_t>, uint16_t>(forward_element<uint16_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<int8_t>::tensor() : precision_(INT8)
	{
	}

	template <>
	inline tensor<int8_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<int8_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT8)
	{
	}

	template <>
	inline tensor<int8_t>::tensor(tensor_shape const& shape, int8_t default_element) :
		data_(make_tensor_data<int8_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(INT8)
	{
		thrust::device_ptr<int8_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<int8_t>, int8_t>(forward_element<int8_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}

	template <>
	inline tensor<uint8_t>::tensor() : precision_(UINT8)
	{
	}

	template <>
	inline tensor<uint8_t>::tensor(tensor_shape const& shape) :
		data_(make_tensor_data<uint8_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT8)
	{
	}

	template <>
	inline tensor<uint8_t>::tensor(tensor_shape const& shape, uint8_t default_element) :
		data_(make_tensor_data<uint8_t>(tensor_shape_total_size(shape))), shape_(shape), precision_(UINT8)
	{
		thrust::device_ptr<uint8_t> dev_ptr(data_.get());
		auto total_size = tensor_shape_total_size(shape_);
		if (should_sanitize())
			thrust::transform(dev_ptr, dev_ptr + total_size, dev_ptr, unary_sanitize<forward_element<uint8_t>, uint8_t>(forward_element<uint8_t>(default_element), precision_.exp_bits, precision_.frac_bits));
		else
			thrust::fill(dev_ptr, dev_ptr + total_size, default_element);
	}
}