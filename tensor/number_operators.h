#pragma once

namespace transprecision_floating_point
{

	template<typename T, typename U>
	struct comparison_operators
	{
		__host__ __device__ static bool equal_to(T lhs, U rhs) { return lhs == rhs; }
		__host__ __device__ static bool not_equal_to(T lhs, U rhs) { return !equal_to(lhs, rhs); }
		__host__ __device__ static bool less(T lhs, U rhs) { return lhs < rhs; }
		__host__ __device__ static bool greater(T lhs, U rhs) { return less(rhs, lhs); }
		__host__ __device__ static bool less_equal(T lhs, U rhs) { return !greater(lhs, rhs); }
		__host__ __device__ static bool greater_equal(T lhs, U rhs) { return !less(lhs, rhs); }
	};

	template<typename T, typename U, typename R>
	struct arithmetic_operators
	{
		__host__ __device__ static R add(T lhs, U rhs) { return lhs + rhs; }
		__host__ __device__ static R sub(T lhs, U rhs) { return lhs - rhs; }
		__host__ __device__ static R mul(T lhs, U rhs) { return lhs * rhs; }
		__host__ __device__ static R div(T lhs, U rhs) { return lhs / rhs; }
		template<typename P>
		__host__ __device__ static R fma(P x, T y, U z) { return R(fmaf(float(x), float(y), float(z))); }
	};
	;
	template<typename T>
	struct comparison_operators<T, half>
	{
		__host__ __device__ static bool equal_to(T lhs, half rhs) { return lhs == float(rhs); }
		__host__ __device__ static bool less(T lhs, half rhs) { return lhs < float(rhs); }
	};

	template<typename U>
	struct comparison_operators<half, U>
	{
		__host__ __device__ static bool equal_to(half lhs, U rhs) { return float(lhs) == rhs; }
		__host__ __device__ static bool less(half lhs, U rhs) { return float(lhs) < rhs; }
	};

	template<>
	struct comparison_operators<half, half>
	{
		__host__ __device__ static bool equal_to(half lhs, half rhs) { return float(lhs) == float(rhs); }
		__host__ __device__ static bool less(half lhs, half rhs) { return float(lhs) < float(rhs); }
	};

	template<typename T, typename R>
	struct arithmetic_operators<T, half, R>
	{
		__host__ __device__ static R add(T lhs, half rhs) { return R(lhs + float(rhs)); }
		__host__ __device__ static R sub(T lhs, half rhs) { return R(lhs - float(rhs)); }
		__host__ __device__ static R mul(T lhs, half rhs) { return R(lhs * float(rhs)); }
		__host__ __device__ static R div(T lhs, half rhs) { return R(lhs / float(rhs)); }
	};

	template<typename U, typename R>
	struct arithmetic_operators<half, U, R>
	{
		__host__ __device__ static R add(half lhs, U rhs) { return R(float(lhs) + rhs); }
		__host__ __device__ static R sub(half lhs, U rhs) { return R(float(lhs) - rhs); }
		__host__ __device__ static R mul(half lhs, U rhs) { return R(float(lhs) * rhs); }
		__host__ __device__ static R div(half lhs, U rhs) { return R(float(lhs) / rhs); }
	};

	template<typename R>
	struct arithmetic_operators<half, half, R>
	{
		__host__ __device__ static R add(half lhs, half rhs) { return R(float(lhs) + float(rhs)); }
		__host__ __device__ static R sub(half lhs, half rhs) { return R(float(lhs) - float(rhs)); }
		__host__ __device__ static R mul(half lhs, half rhs) { return R(float(lhs) * float(rhs)); }
		__host__ __device__ static R div(half lhs, half rhs) { return R(float(lhs) / float(rhs)); }
	};
}