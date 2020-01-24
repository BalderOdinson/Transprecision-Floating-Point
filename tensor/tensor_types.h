#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_error_defines.h"
#include <numeric>

namespace transprecision_floating_point
{
	using tensor_shape = std::vector<size_t>;
	using tensor_index = tensor_shape;

	inline size_t tensor_shape_dimension(tensor_shape const& shape) { return shape.size(); }
	inline size_t tensor_shape_total_size(tensor_shape const& shape) { return  std::accumulate(shape.begin(), shape.end(), size_t(1), [](size_t total, size_t current)
	{
		return total * current;
	}); }
	inline size_t tensor_index_calculate(tensor_shape const& shape, tensor_index const& index)
	{
		auto stride = std::accumulate(shape.begin(), shape.end(), size_t(1), [](size_t total, size_t current) { return total * current; });
		size_t result = 0;
		for (size_t i = 0; i < shape.size(); ++i)
		{
			stride /= shape[i];
			result += index[i] * stride;
		}
		return result;
	}
	inline tensor_shape tensor_index_calculate(tensor_shape const& shape, size_t index)
	{
		auto stride = std::accumulate(shape.begin(), shape.end(), size_t(1), [](size_t total, size_t current) { return total * current; });
		tensor_shape result(shape.size());
		size_t i = 0;
		for (; i < shape.size() - 1; ++i)
		{
			stride /= shape[i];
			result[i] = (index / stride) % shape[i];
		}
		result[i] = index % shape[i];
		return result;
	}

	struct tensor_precision
	{
		uint_fast8_t exp_bits;
		uint_fast8_t frac_bits;
	};

	inline bool operator==(tensor_precision const& lhs, tensor_precision const& rhs) { return lhs.exp_bits == rhs.exp_bits && lhs.frac_bits == rhs.frac_bits; }
	inline bool operator!=(tensor_precision const& lhs, tensor_precision const& rhs) { return !operator==(lhs, rhs); }
	inline bool operator< (tensor_precision const& lhs, tensor_precision const& rhs) { return lhs.exp_bits == rhs.exp_bits ? lhs.frac_bits < rhs.frac_bits : lhs.exp_bits < rhs.exp_bits; }
	inline bool operator> (tensor_precision const& lhs, tensor_precision const& rhs) { return  operator< (rhs, lhs); }
	inline bool operator<=(tensor_precision const& lhs, tensor_precision const& rhs) { return !operator> (lhs, rhs); }
	inline bool operator>=(tensor_precision const& lhs, tensor_precision const& rhs) { return !operator< (lhs, rhs); }

	template<typename T>
	using tensor_data = std::unique_ptr<T[], std::function<void(T*)>>;

	template<typename T>
	tensor_data<T> make_tensor_data(size_t size)
	{
		static auto deleter = [](T* data)
		{
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			CHECK_CUDA_ERROR(cudaFree(data));
		};

		T* data;
		CHECK_CUDA_ERROR(cudaMallocManaged(&data, size * sizeof(T)));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		return tensor_data<T>(data, deleter);
	}
}
