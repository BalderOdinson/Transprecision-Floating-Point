#pragma once
#include "cuda_procedures_helpers.h"
#include "broadcasting_helper.h"


namespace transprecision_floating_point
{
	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	template<typename T>
	__global__ void _transpose_kernel(T* src, T* dst, size_t n, size_t m);
	template<typename T>
	void _transpose(T* src, T* dst, tensor_shape const& shape);

	template<typename T, typename U, typename R>
	__global__ void _matrix_multiply_kernel(T* first, U* second, R* dst, size_t n, size_t m, size_t p);
	template<typename T, typename U, typename R>
	void _matrix_multiply(tensor_data<T> const& first, tensor_data<U> const& second, tensor_data<R>& dst, tensor_shape const& first_shape, tensor_shape const& second_shape);
	template<typename T, typename U, typename R>
	__global__ void _matrix_multiply_kernel(T* first, U* second, R* dst, size_t n, size_t m, size_t p, uint_fast8_t exp_bits, uint_fast8_t frac_bits);
	template<typename T, typename U, typename R>
	void _matrix_multiply(tensor_data<T> const& first, tensor_data<U> const& second, tensor_data<R>& dst, tensor_shape const& first_shape, tensor_shape const& second_shape, tensor_precision const& precision);

	template<typename T, typename Function>
	__global__ void _produce_kernel(T* first, T* second, Function produce, size_t n, size_t m, uint_fast8_t broadcast_axis = 3, size_t broadcast_size = 0);
	template<typename T, typename Function>
	void _produce(tensor_data<T>& first, tensor_data<T> const& second, Function produce, tensor_shape const& shape_first, tensor_shape const& shape_second);
	template<typename T, typename Function>
	__global__ void _produce_kernel(T* first, T* second, Function produce, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits, uint_fast8_t broadcast_axis = 3, size_t broadcast_size = 0);
	template<typename T, typename Function>
	void _produce(tensor_data<T> const& first, tensor_data<T> const& second, Function produce, tensor_shape const& shape_first, tensor_shape const& shape_second, tensor_precision const& precision);

	template<typename T, typename Function>
	__global__ void _apply_kernel(T* dst, Function apply, size_t n, size_t m);
	template<typename T, typename Function>
	void _apply(tensor_data<T>& dst, Function apply, tensor_shape const& shape);
	template<typename T, typename Function>
	__global__ void _apply_kernel(T* dst, Function apply, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits);
	template<typename T, typename Function>
	void _apply(tensor_data<T>& dst, Function apply, tensor_shape const& shape, tensor_precision const& precision);

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */

	template<typename T, typename U, typename R>
	__global__ void _matrix_multiply_kernel(T* first, U* second, R* dst, size_t n, size_t m, size_t p)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < p)
		{
			R dot = R(0);
			for (size_t k = 0; k < m; ++k)
				dot = arithmetic_operators<T, U, R>::add(dot, arithmetic_operators<T, U, R>::mul(first[i * m + k], second[j * m + k]));

			dst[i * p + j] = dot;
		}

	}

	template<typename T, typename U, typename R>
	void _matrix_multiply(tensor_data<T> const& first, tensor_data<U> const& second, tensor_data<R>& dst, tensor_shape const& first_shape, tensor_shape const& second_shape)
	{
		auto const total_size_second = tensor_shape_total_size(second_shape);
		auto n = first_shape.front();
		auto m = second_shape.front();
		auto p = total_size_second / m;
		auto tensor_dim = _calculate_block_size(tensor_shape{n,p});
		_matrix_multiply_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (first.get(), second.get(), dst.get(), n, m, p);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	template<typename T, typename U, typename R>
	__global__ void _matrix_multiply_kernel(T* first, U* second, R* dst, size_t n, size_t m, size_t p, uint_fast8_t exp_bits, uint_fast8_t frac_bits)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < p)
		{
			T dot = T(0);
			for (size_t k = 0; k < m; ++k)
				dot = arithmetic_operators<T, U, R>::add(dot, arithmetic_operators<T, U, R>::mul(first[i * m + k], second[j * m + k]));

			dst[i * p + j] = sanitize(exp_bits,frac_bits, dot);
		}

	}

	template<typename T, typename U, typename R>
	void _matrix_multiply(tensor_data<T> const& first, tensor_data<U> const& second, tensor_data<R>& dst, tensor_shape const& first_shape, tensor_shape const& second_shape, tensor_precision const& precision)
	{
		auto const total_size_second = tensor_shape_total_size(second_shape);
		auto n = first_shape.front();
		auto m = second_shape.front();
		auto p = total_size_second / m;
		auto tensor_dim = _calculate_block_size(tensor_shape{ n,p });
		_matrix_multiply_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (first.get(), second.get(), dst.get(), n, m, p, precision.exp_bits, precision.frac_bits);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	template<typename T>
	__global__ void _transpose_kernel(T* src, T* dst, size_t n, size_t m)
	{
		size_t const id = blockIdx.x*blockDim.x + threadIdx.x;

		if (id < n*m)
		{
			auto const i = id / n;
			auto const j = id % n;
			dst[id] = src[j * m + i];
		}

	}

	template<typename T>
	void _transpose(T* src, T* dst, tensor_shape const& shape)
	{
		auto tensor_dim = _calculate_block_size(shape);
		dim3 threads_per_block(tensor_dim.n * tensor_dim.m);
		dim3 blocks_per_grid(1);
		if (threads_per_block.x > BLOCK_SIZE)
		{
			threads_per_block.x = BLOCK_SIZE;
			blocks_per_grid.x = ceil(double(tensor_dim.n * tensor_dim.m) / double(threads_per_block.x));
		}
		_transpose_kernel << <blocks_per_grid, threads_per_block >> > (src, dst, tensor_dim.n, tensor_dim.m);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	template<typename T>
	__global__ void _broadcast_kernel(T* first, T* second, size_t n, size_t m, uint_fast8_t broadcast_axis, size_t broadcast_size)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
		{
			if (broadcast_axis == 0)
				first[i * m + j] = second[i % broadcast_size];
			if (broadcast_axis == 1)
				first[i * m + j] = second[j % broadcast_size];
		}
	}

	template<typename T>
	void _broadcast(T* dst, T* src, tensor_shape const& shape_first, tensor_shape const& shape_second)
	{
		broadcasting_helper broadcasting(shape_second, shape_first);
		if (!broadcasting.can_broadcast())
			throw std::runtime_error("Invalid shapes - " + to_string(shape_first) + " and " + to_string(shape_second) + "!");
		auto tensor_dim = broadcasting.axis() == 0 ? _calculate_block_size_inverse(shape_first) : _calculate_block_size(shape_first);
		_broadcast_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (
			dst, src, tensor_dim.n, tensor_dim.m, broadcasting.axis(), broadcasting.tensor_size());
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	template<typename T, typename Function>
	__global__ void _produce_kernel(T* first, T* second, Function produce, size_t n, size_t m, uint_fast8_t broadcast_axis, size_t broadcast_size)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
		{
			if (broadcast_axis == 0)
				first[i * m + j] = T(produce(first[i * m + j], second[i % broadcast_size]));
			else if (broadcast_axis == 1)
				first[i * m + j] = T(produce(first[i * m + j], second[j % broadcast_size]));
			else
				first[i * m + j] = T(produce(first[i * m + j], second[i * m + j]));
		}
	}

	template<typename T, typename Function>
	void _produce(tensor_data<T>& first, tensor_data<T> const& second, Function produce, tensor_shape const& shape_first, tensor_shape const& shape_second)
	{
		if (shape_first == shape_second)
		{
			auto tensor_dim = _calculate_block_size(shape_first);
			_produce_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (
				first.get(), second.get(), produce, tensor_dim.n, tensor_dim.m);
			CHECK_CUDA_ERROR(cudaGetLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
		else
		{
			broadcasting_helper broadcasting(shape_second, shape_first);
			if (!broadcasting.can_broadcast())
				throw std::runtime_error("Invalid shapes - " + to_string(shape_first) + " and " + to_string(shape_second) + "!");
			auto tensor_dim = broadcasting.axis() == 0 ? _calculate_block_size_inverse(shape_first) : _calculate_block_size(shape_first);
			_produce_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (
				first.get(), second.get(), produce, tensor_dim.n, tensor_dim.m, broadcasting.axis(), broadcasting.tensor_size());
			CHECK_CUDA_ERROR(cudaGetLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
	}

	template<typename T, typename Function>
	__global__ void _produce_kernel(T* first, T* second, Function produce, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits, uint_fast8_t broadcast_axis, size_t broadcast_size)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
		{
			if (broadcast_axis == 0)
				first[i * m + j] = sanitize(exp_bits, frac_bits, T(produce(first[i * m + j], second[i % broadcast_size])));
			else if (broadcast_axis == 1)
				first[i * m + j] = sanitize(exp_bits, frac_bits, T(produce(first[i * m + j], second[j % broadcast_size])));
			else
				first[i * m + j] = sanitize(exp_bits, frac_bits, T(produce(first[i * m + j], second[i * m + j])));
		}
	}

	template<typename T, typename Function>
	void _produce(tensor_data<T> const& first, tensor_data<T> const& second, Function produce, tensor_shape const& shape_first, tensor_shape const& shape_second, tensor_precision const& precision)
	{
		if (shape_first == shape_second)
		{
			auto tensor_dim = _calculate_block_size(shape_first);
			_produce_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (
				first.get(), second.get(), produce, tensor_dim.n, tensor_dim.m, 3, precision.exp_bits, precision.frac_bits);
			CHECK_CUDA_ERROR(cudaGetLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
		else
		{
			broadcasting_helper broadcasting(shape_second, shape_first);
			if (!broadcasting.can_broadcast())
				throw std::runtime_error("Invalid shapes - " + to_string(shape_first) + " and " + to_string(shape_second) + "!");
			auto tensor_dim = broadcasting.axis() == 0 ? _calculate_block_size_inverse(shape_first) : _calculate_block_size(shape_first);
			_produce_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (
				first.get(), second.get(), produce, tensor_dim.n, tensor_dim.m, precision.exp_bits, precision.frac_bits, broadcasting.axis(), broadcasting.tensor_size());
			CHECK_CUDA_ERROR(cudaGetLastError());
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
	}

	template<typename T, typename Function>
	__global__ void _apply_kernel(T* dst, Function apply, size_t n, size_t m)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
			dst[i * m + j] = apply(dst[i * m + j]);
	}

	template<typename T, typename Function>
	void _apply(tensor_data<T>& dst, Function apply, tensor_shape const& shape)
	{
		auto tensor_dim = _calculate_block_size(shape);
		_apply_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (dst.get(), apply, tensor_dim.n, tensor_dim.m);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	template<typename T, typename Function>
	__global__ void _apply_kernel(T* dst, Function apply, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
			dst[i * m + j] = sanitize(exp_bits, frac_bits, apply(dst[i * m + j]));
	}

	template<typename T, typename Function>
	void _apply(tensor_data<T>& dst, Function apply, tensor_shape const& shape, tensor_precision const& precision)
	{
		auto tensor_dim = _calculate_block_size(shape);
		_apply_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (dst.get(), apply, tensor_dim.n, tensor_dim.m, precision.exp_bits, precision.frac_bits);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */
}
