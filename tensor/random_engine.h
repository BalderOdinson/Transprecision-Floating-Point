#pragma once
#include <curand.h>
#include <random>

namespace transprecision_floating_point
{
	struct random_engine
	{
		template<typename Distribution>
		static void generate(float* a, size_t size, Distribution distribution);
		static void set_seed(size_t seed);
		static void destroy();

	private:
		static void init();
		static bool is_rng_init_;
		static curandGenerator_t rng_;
	};

	bool random_engine::is_rng_init_ = false;
	curandGenerator_t random_engine::rng_;

	template <typename Distribution>
	void random_engine::generate(float* a, size_t size, Distribution distribution)
	{
		init();
		CHECK_CURAND_ERROR(distribution(random_engine::rng_, a, size));
	}

	inline void random_engine::set_seed(size_t seed)
	{
		init();
		CHECK_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(random_engine::rng_, seed));
	}

	inline void random_engine::destroy()
	{
		CHECK_CURAND_ERROR(curandDestroyGenerator(rng_));
	}

	inline void random_engine::init()
	{
		if(random_engine::is_rng_init_) return;
		CHECK_CURAND_ERROR(curandCreateGenerator(&random_engine::rng_, CURAND_RNG_PSEUDO_DEFAULT));
		CHECK_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(random_engine::rng_, std::random_device{}()));
		random_engine::is_rng_init_ = true;
	}

	struct uniform_distribution
	{
		curandStatus_t operator()(curandGenerator_t rng, float* a, size_t size) const
		{
			return curandGenerateUniform(rng, a, size);
		}
	};

	struct normal_distribution
	{
		normal_distribution(float mean = 0, float stddev = 1) : mean_(mean), stddev_(stddev) {}

		curandStatus_t operator()(curandGenerator_t rng, float* a, size_t size) const
		{
			return curandGenerateNormal(rng, a, size % 2 == 0 ? size : size + 1, mean_, stddev_);
		}

	private:
		float mean_;
		float stddev_;
	};

	struct normal_scaled_distribution
	{
		normal_scaled_distribution(float inputs) : inputs_(inputs) {}

		curandStatus_t operator()(curandGenerator_t rng, float* a, size_t size) const
		{
			return curandGenerateNormal(rng, a, size % 2 == 0 ? size : size + 1, 0, sqrtf(2 / inputs_));
		}

	private:
		float inputs_;
	};

	struct xavier_distribution
	{
		xavier_distribution(float inputs, float outputs) : inputs_(inputs), outputs_(outputs) {}

		curandStatus_t operator()(curandGenerator_t rng, float* a, size_t size) const
		{
			return curandGenerateNormal(rng, a, size % 2 == 0 ? size : size + 1, 0, sqrtf(1/(inputs_*outputs_)));
		}

	private:
		float inputs_;
		float outputs_;
	};

}
