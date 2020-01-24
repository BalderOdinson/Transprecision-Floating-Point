#pragma once
#include <omp.h>
#include <random>

namespace transprecision_floating_point::simple_blas
{
	template<typename Dist>
	float thread_safe_rand(Dist distribution)
	{
		static thread_local std::mt19937 generator(/*std::random_device{}() +*/ omp_get_thread_num());
		return distribution(generator);
	}
}