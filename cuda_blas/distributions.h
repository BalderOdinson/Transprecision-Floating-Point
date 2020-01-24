#pragma once
#include "curand_kernel.h"

namespace transprecision_floating_point
{
	namespace cuda_blas
	{
		struct normal_distribution
		{
			normal_distribution(float mean = 0, float std_dev = 1);
			__host__ __device__ normal_distribution(normal_distribution const& other);
			__device__ float operator()(curandState_t& state) const;

		private:
			float mean_;
			float std_dev_;
		};

		struct uniform_distribution
		{
			uniform_distribution(float min = 0, float max = 1);
			__host__ __device__ uniform_distribution(uniform_distribution const& other);
			__device__ float operator()(curandState_t& state) const;

		private:
			float min_;
			float max_;
		};

		struct variance_normal_initializer
		{
			variance_normal_initializer(float fan_in);
			__host__ __device__ variance_normal_initializer(variance_normal_initializer const& other);
			__device__ float operator()(curandState_t& state) const;

		private:
			float fan_in_;
		};

		inline normal_distribution::normal_distribution(float mean, float std_dev) : mean_(mean), std_dev_(std_dev)
		{
		}

		inline normal_distribution::normal_distribution(normal_distribution const& other) : mean_(other.mean_), std_dev_(other.std_dev_)
		{
		}

		inline float normal_distribution::operator()(curandState_t& state) const
		{
			return curand_normal(&state) * std_dev_ + mean_;
		}

		inline uniform_distribution::uniform_distribution(float min, float max) : min_(min), max_(max)
		{
		}

		inline uniform_distribution::uniform_distribution(uniform_distribution const& other): min_(other.min_), max_(other.max_)
		{
		}

		inline float uniform_distribution::operator()(curandState_t& state) const
		{
			return curand_uniform(&state) * (max_ - min_) + min_;
		}

		inline variance_normal_initializer::variance_normal_initializer(float fan_in) : fan_in_(fan_in)
		{
		}

		inline variance_normal_initializer::variance_normal_initializer(variance_normal_initializer const& other): fan_in_(other.fan_in_)
		{
		}

		inline float variance_normal_initializer::operator()(curandState_t& state) const
		{
			auto const sigma = sqrtf(2 / fan_in_);
			return curand_normal(&state) * 2 * sigma;
		}
	}
}
