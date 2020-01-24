#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tensor_types.h"
#include <random>

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_2D 32

namespace transprecision_floating_point
{
	namespace cuda_blas
	{
		static size_t get_random_seed()
		{
			static size_t seed = 1;
			return seed++;
		}

		inline void _process_cuda_error(cudaError_t error)
		{
			if (error != cudaSuccess)
				throw std::runtime_error(cudaGetErrorString(error));
		}

		inline std::pair<dim3, dim3> _get_2d_block_size(size_t n, size_t m)
		{
			dim3 threads_per_block(n, m);
			dim3 blocks_per_grid(1, 1);
			if (n*m > BLOCK_SIZE_2D)
			{
				threads_per_block.x = BLOCK_SIZE_2D;
				threads_per_block.y = BLOCK_SIZE_2D;
				blocks_per_grid.x = ceil(double(n) / double(threads_per_block.x));
				blocks_per_grid.y = ceil(double(m) / double(threads_per_block.y));
			}

			return { blocks_per_grid,threads_per_block };
		}

		template<typename Type>
		__global__ void _copy_kernel(Type* src, Type* dst, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = src[i * m + j];
		}

		template<typename Type>
		void _copy(tensor_data<Type> const& src, tensor_data<Type>& dst, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_copy_kernel << <kernel_arguments.first, kernel_arguments.second >> > (src.get(), dst.get(), shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type>
		void _copy(Type* src, Type* dst, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_copy_kernel << <kernel_arguments.first, kernel_arguments.second >> > (src, dst, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type>
		__global__ void _initialize_kernel(Type* dst, Type element, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = element;
		}

		template<typename Type>
		void _initialize(tensor_data<Type>& dst, Type element, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_initialize_kernel << <kernel_arguments.first, kernel_arguments.second >> > (dst.get(), element, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename TOne, typename TTwo>
		__global__ void _cast_kernel(TOne* src, TTwo* dst, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = TTwo(src[i * m + j]);
		}

		template<typename TOne, typename TTwo>
		void _cast(tensor_data<TOne> const& src, tensor_data<TTwo>& dst, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_cast_kernel << <kernel_arguments.first, kernel_arguments.second >> > (src.get(), dst.get(), shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type>
		__global__ void _copy_from_list_kernel(Type* src, Type* dst, size_t n, size_t m, size_t copy_line)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = src[(i % copy_line) * m + j];
		}

		template<typename Type>
		void _copy_from_list(std::initializer_list<Type> const& src, tensor_data<Type>& dst, tensor_shape const& shape, size_t copy_line)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			Type* s;
			_process_cuda_error(cudaMalloc(&s, src.size() * sizeof(Type)));
			_process_cuda_error(cudaMemcpy(s, data(src), src.size() * sizeof(Type), cudaMemcpyHostToDevice));
			_copy_from_list_kernel << <kernel_arguments.first, kernel_arguments.second >> > (s, dst.get(), shape.first, shape.second, copy_line);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
			_process_cuda_error(cudaFree(s));
		}

		__global__ inline void _initialize_curand_state(curandState_t* global_states, size_t seed, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;
			auto id = i * m + j;

			if (i < n && j < m)
				curand_init(seed, id, 0, &global_states[id]);
		}

		template<typename Type, typename Distribution>
		__global__ void _random_initialize_kernel(Type* dst, Distribution distribution, curandState_t* global_states, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = distribution(global_states[i * m + j]);
		}

		template<typename Type, typename Distribution>
		void _random_initialize(tensor_data<Type>& dst, Distribution& distribution, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			curandState* states;
			_process_cuda_error(cudaMalloc(&states, shape.first * shape.second * sizeof(curandState)));
			_initialize_curand_state << <kernel_arguments.first, kernel_arguments.second >> > (states, get_random_seed(), shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
			_random_initialize_kernel << <kernel_arguments.first, kernel_arguments.second >> > (dst.get(), distribution, states, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
			_process_cuda_error(cudaFree(states));
		}

		template<typename Type>
		__global__ void _initialize_diagonal_kernel(Type* src, Type* dst, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = i == j ? (src ? src[i] : Type(1)) : Type(0);
		}

		template<typename Type>
		void _initialize_diagonal(std::initializer_list<Type> const& src, tensor_data<Type>& dst, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			if (src.size() != 0)
			{
				Type* s;
				_process_cuda_error(cudaMalloc(&s, src.size() * sizeof(Type)));
				_process_cuda_error(cudaMemcpy(s, data(src), src.size() * sizeof(Type), cudaMemcpyHostToDevice));
				_initialize_diagonal_kernel << <kernel_arguments.first, kernel_arguments.second >> > (s, dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
				_process_cuda_error(cudaFree(s));
			}
			else
			{
				Type* s = nullptr;
				_initialize_diagonal_kernel << <kernel_arguments.first, kernel_arguments.second >> > (s, dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
		}

		template<typename Type>
		__global__ void _transpose_kernel(Type* src, Type* dst, size_t n, size_t m)
		{
			size_t const id = blockIdx.x*blockDim.x + threadIdx.x;

			if (id < n*m)
			{
				auto const i = id / n;
				auto const j = id % n;
				dst[id] = src[j * m + i];
			}

		}

		template<typename Type>
		void _transpose(tensor_data<Type> const& src, tensor_data<Type>& dst, tensor_shape const& shape)
		{
			dim3 threads_per_block(shape.first * shape.second);
			dim3 blocks_per_grid(1);
			if (threads_per_block.x > BLOCK_SIZE)
			{
				threads_per_block.x = BLOCK_SIZE;
				blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
			}
			_transpose_kernel << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type>
		__global__ void _matrix_multiply_kernel(Type* first, Type* second, Type* dst, size_t n, size_t m, size_t p)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < p)
			{
				Type dot = 0;
				for (size_t k = 0; k < m; ++k)
					dot += first[i * m + k] * second[j * m + k];

				dst[i * p + j] = dot;
			}

		}

		template<typename Type>
		void _matrix_multiply(tensor_data<Type> const& first, tensor_data<Type> const& second, tensor_data<Type>& dst, size_t n, size_t m, size_t p)
		{
			auto kernel_arguments = _get_2d_block_size(n,p);
			_matrix_multiply_kernel << <kernel_arguments.first, kernel_arguments.second >> > (first.get(), second.get(), dst.get(), n, m, p);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type, uint_fast8_t Axis, typename Function>
		__global__ void _aggregate_kernel(Type* src, Type* dst, Function aggregate, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;

			if(Axis == 0 && i < m)
			{
				dst[i] = src[i];
				for (int64_t j = 1; j < n; ++j)
					dst[i] = aggregate(dst[i], src[j * m + i]);
			}
			if(Axis == 1 && i < n)
			{
				dst[i] = src[i * m];
				for (int64_t j = 1; j < m; ++j)
					dst[i] = aggregate(dst[i], src[i * m + j]);
			}
		}

		template<typename Type, uint_fast8_t Axis, typename Function>
		void _aggregate(tensor_data<Type> const& src, tensor_data<Type>& dst, tensor_shape const& shape, Function aggregate)
		{
			if(Axis == 0)
			{
				dim3 threads_per_block(shape.second);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_aggregate_kernel<Type, Axis, Function> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), aggregate, shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
			if(Axis == 1)
			{
				dim3 threads_per_block(shape.first);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_aggregate_kernel<Type, Axis, Function> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), aggregate, shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
		}

		template<typename Type, uint_fast8_t Axis>
		__global__ void _argmax_kernel(Type* src, size_t* dst, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;

			if (Axis == 0 && i < m)
			{
				dst[i] = 0;
				for (size_t j = 1; j < n; ++j)
					dst[i] = float(dst[i]) < src[j * m + i] ? j : dst[i];
			}
			if (Axis == 1 && i < n)
			{
				dst[i] = 0;
				for (size_t j = 1; j < m; ++j)
					dst[i] = float(dst[i]) < src[i * m + j] ? j : dst[i];
			}
		}

		template<typename Type, uint_fast8_t Axis>
		void _argmax(tensor_data<Type> const& src, tensor_data<size_t>& dst, tensor_shape const& shape)
		{
			if (Axis == 0)
			{
				dim3 threads_per_block(shape.second);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_argmax_kernel<Type, Axis> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
			if (Axis == 1)
			{
				dim3 threads_per_block(shape.first);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_argmax_kernel<Type, Axis> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
		}

		template<typename Type, uint_fast8_t Axis, typename Function>
		__global__ void _argmin_kernel(Type* src, size_t* dst, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;

			if (Axis == 0 && i < m)
			{
				dst[i] = 0;
				for (size_t j = 1; j < n; ++j)
					dst[i] = float(dst[i]) > src[j * m + i] ? j : dst[i];
			}
			if (Axis == 1 && i < n)
			{
				dst[i] = 0;
				for (size_t j = 1; j < m; ++j)
					dst[i] = float(dst[i]) > src[i * m + j] ? j : dst[i];
			}
		}

		template<typename Type, uint_fast8_t Axis>
		void _argmin(tensor_data<Type> const& src, tensor_data<size_t>& dst, tensor_shape const& shape)
		{
			if (Axis == 0)
			{
				dim3 threads_per_block(shape.second);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_argmin_kernel<Type, Axis> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
			if (Axis == 1)
			{
				dim3 threads_per_block(shape.first);
				dim3 blocks_per_grid(1);
				if (threads_per_block.x > BLOCK_SIZE)
				{
					threads_per_block.x = BLOCK_SIZE;
					blocks_per_grid.x = ceil(double(shape.first * shape.second) / double(threads_per_block.x));
				}
				_argmin_kernel<Type, Axis> << <blocks_per_grid, threads_per_block >> > (src.get(), dst.get(), shape.first, shape.second);
				_process_cuda_error(cudaGetLastError());
				_process_cuda_error(cudaDeviceSynchronize());
			}
		}

		template<typename Type, typename Function>
		__global__ void _apply_kernel(Type* dst, Function apply, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = apply(dst[i * m + j]);
		}

		template<typename Type, typename Function>
		void _apply(tensor_data<Type>& dst, Function apply, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_apply_kernel << <kernel_arguments.first, kernel_arguments.second >> > (dst.get(), apply, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type, typename Function>
		__global__ void _produce_kernel(Type* first, Type* second, Type *dst, Function produce, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
				dst[i * m + j] = Type(produce(first[i * m + j], second[i * m + j]));
		}

		template<typename Type, typename Function>
		void _produce(tensor_data<Type> const& first, tensor_data<Type> const& second, tensor_data<Type>& dst, Function produce, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_produce_kernel << <kernel_arguments.first, kernel_arguments.second >> > (first.get(), second.get(), dst.get(), produce, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type, typename Function, uint_fast8_t Axis>
		__global__ void _produce_kernel(Type* first, Type* second, Type *dst, Function produce, size_t n, size_t m)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n && j < m)
			{
				if(Axis == 0)
					dst[i * m + j] = Type(produce(first[i * m + j], second[i]));
				if(Axis == 1)
					dst[i * m + j] = Type(produce(first[i * m + j], second[j]));
			}
		}

		template<typename Type, typename Function, uint_fast8_t Axis>
		void _produce(tensor_data<Type> const& first, tensor_data<Type> const& second, tensor_data<Type>& dst, Function produce, tensor_shape const& shape)
		{
			auto kernel_arguments = _get_2d_block_size(shape.first, shape.second);
			_produce_kernel<Type,Function,Axis> << <kernel_arguments.first, kernel_arguments.second >> > (first.get(), second.get(), dst.get(), produce, shape.first, shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
		}

		template<typename Type>
		__global__ void _permutate_kernel(Type* first, Type* second, Type* dest_first, Type* dest_second, size_t* indices, size_t n, size_t m, size_t p)
		{
			size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
			size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

			if (i < n)
			{
				if(j < m)
					dest_first[i * m + j] = first[indices[i] * m + j];
				if(j < p)
					dest_second[i * p + j] = second[indices[i] * p + j];
			}
		}

		template<typename Type>
		void _permutate(tensor_data<Type> const& first, tensor_data<Type> const& second, tensor_data<Type>& dest_first, tensor_data<Type>& dest_second, tensor_shape const& first_shape, tensor_shape const& second_shape)
		{
			auto kernel_arguments = _get_2d_block_size(first_shape.first, std::max(first_shape.second, second_shape.second));

			std::vector<size_t> indices(first_shape.first);
			size_t idx = 0;
			std::generate(indices.begin(), indices.end(), [&idx]() {  return idx++; });
			std::mt19937 g(get_random_seed());
			std::shuffle(indices.begin(), indices.end(), g);
			size_t* s;
			_process_cuda_error(cudaMalloc(&s, indices.size() * sizeof(size_t)));
			_process_cuda_error(cudaMemcpy(s, indices.data(), indices.size() * sizeof(size_t), cudaMemcpyHostToDevice));
			_permutate_kernel << <kernel_arguments.first, kernel_arguments.second >> > (first.get(), second.get(), dest_first.get(), dest_second.get(), s, first_shape.first, first_shape.second, second_shape.second);
			_process_cuda_error(cudaGetLastError());
			_process_cuda_error(cudaDeviceSynchronize());
			_process_cuda_error(cudaFree(s));
		}
	}
}
