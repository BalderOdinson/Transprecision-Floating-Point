#pragma once
#include "cuda_procedures_helpers.h"
#include <fenv.h>
#include <bitset>
#pragma STDC FENV_ACCESS ON

#define UINT_C UINT32_C
#define MASK_FRAC (UINT32_C(0x007FFFFF))
#define MASK_FRAC_MSB (UINT32_C(0x00800000))
#define MASK_FRAC_EXCEPT_MSB (UINT32_C(0x003FFFFF))
#define SMALLEST_NORM_POS (0x00800000)
#define SMALLEST_NORM_NEG (0x80800000)
#define INF_EXP (0xFF)
#define BIAS (127)
#define NUM_BITS (32)
#define NUM_BITS_EXP (8)
#define NUM_BITS_FRAC (23)
typedef int32_t int_t;
typedef uint32_t uint_t;

#define SIGN( a ) ((bool) ((uint_t) (a)>>(NUM_BITS-1)))
#define EXPONENT( a ) ((int_fast16_t) ((uint_t) (a)>>NUM_BITS_FRAC) & INF_EXP)
#define PACK( sign, exp, sig ) ((uint_t) (((uint_t) (sign)<<(NUM_BITS-1)) + ((uint_t) (exp)<<NUM_BITS_FRAC) + (sig)))

#define CAST_TO_INT(d) (*((int_t *)(&(d))))
#define CAST_TO_UINT(d) (*((uint_t *)(&(d))))
#define CAST_TO_FP(d) (*((float *)(&(d))))


namespace transprecision_floating_point
{

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	__host__ __device__ bool sign(float value);
	__host__ __device__ int_fast16_t exponent(uint_fast8_t exp_bits, float value);
	__host__ __device__ int_fast16_t bias(uint_fast8_t exp_bits);
	__host__ __device__ int_fast16_t inf_exp(uint_fast8_t exp_bits);
	__host__ __device__ bool round_bit(uint_fast8_t frac_bits, float value, int_fast16_t exp);
	__host__ __device__ bool sticky_bit(uint_fast8_t frac_bits, float value, int_fast16_t exp);
	__host__ __device__ uint_t denorm_frac(uint_fast8_t frac_bits, float value, int_fast16_t exp);
	__host__ __device__ uint_t frac(uint_fast8_t frac_bits, float value);
	__host__ __device__ bool nearest_rounding(uint_fast8_t frac_bits, float value, int_fast16_t exp);
	__host__ __device__ uint_t denorm_pack(uint_fast8_t frac_bits, bool sign, uint_t frac);
	__host__ __device__ uint_t pack(uint_fast8_t exp_bits, uint_fast8_t frac_bits, bool sign, int_fast16_t exp, uint_t frac);
	__host__ __device__ int_t rounding_value(uint_fast8_t exp_bits, uint_fast8_t frac_bits, float value, int_fast16_t exp, bool sign);
	__host__ __device__ bool inf_rounding(uint_fast8_t exp_bits, uint_fast8_t frac_bits, float value, int_fast16_t exp, bool sign, bool plus);
	template<typename T>
	__host__ __device__ T sanitize(uint_fast8_t exp_bits, uint_fast8_t frac_bits, T value);

	template<typename T>
	__global__ void _sanitize_kernel(T* data, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits);
	template<typename T>
	void _sanitize(tensor_data<T> const& data, tensor_shape const& shape, tensor_precision const& precision);

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */

	inline bool sign(float value)
	{
		return (CAST_TO_INT(value)) >> (NUM_BITS - 1);
	}

	inline int_fast16_t exponent(uint_fast8_t exp_bits, float value)
	{
		auto const a_exp = EXPONENT(CAST_TO_INT(value));

		auto const bias = transprecision_floating_point::bias(exp_bits);

		if (a_exp == 0 || a_exp == INF_EXP)
			return a_exp;

		return (a_exp - BIAS) + bias;
	}

	inline int_fast16_t bias(uint_fast8_t exp_bits)
	{
		return int_fast16_t((int_fast16_t(1) << (exp_bits - 1)) - 1);
	}

	inline int_fast16_t inf_exp(uint_fast8_t exp_bits)
	{
		return int_fast16_t((int_fast16_t(1) << exp_bits) - 1);
	}

	inline bool round_bit(uint_fast8_t frac_bits, float value, int_fast16_t exp)
	{
		if (exp <= 0 && EXPONENT(CAST_TO_INT(value)) != 0)
		{
			auto const shift = (-exp + 1);
			uint_t denorm = 0;
			if (shift < NUM_BITS)
				denorm = ((CAST_TO_INT(value) & MASK_FRAC | MASK_FRAC_MSB)) >> shift;
			return denorm & (UINT_C(0x1) << (NUM_BITS_FRAC - frac_bits - 1));
		}
		return CAST_TO_INT(value) & (UINT_C(0x1) << (NUM_BITS_FRAC - frac_bits - 1));
	}

	inline bool sticky_bit(uint_fast8_t frac_bits, float value, int_fast16_t exp)
	{
		if (exp <= 0 && EXPONENT(CAST_TO_INT(value)) != 0)
		{
			auto const shift = (-exp + 1);
			uint_t denorm = 0;
			if (shift < NUM_BITS)
				denorm = ((CAST_TO_INT(value) & MASK_FRAC) | MASK_FRAC_MSB) >> shift;
			return (denorm & (MASK_FRAC >> (frac_bits + 1))) ||
				(((denorm & MASK_FRAC) == 0) && (CAST_TO_INT(value) != 0));
		}
		return CAST_TO_INT(value) & (MASK_FRAC >> (frac_bits + 1));
	}

	inline uint_t denorm_frac(uint_fast8_t frac_bits, float value, int_fast16_t exp)
	{
		if (EXPONENT(CAST_TO_INT(value)) == 0) // Denormalized backend value
			return (CAST_TO_INT(value) & MASK_FRAC) >> (NUM_BITS_FRAC - frac_bits);

		// Denormalized target value (in normalized backend value)
		unsigned short const shift = NUM_BITS_FRAC - frac_bits - exp + 1;
		if (shift >= NUM_BITS) return false;
		return (((CAST_TO_INT(value) & MASK_FRAC) | MASK_FRAC_MSB) >> shift);
	}

	inline uint_t frac(uint_fast8_t frac_bits, float value)
	{
		return (CAST_TO_INT(value) & MASK_FRAC) >> (NUM_BITS_FRAC - frac_bits);
	}

	inline bool nearest_rounding(uint_fast8_t frac_bits, float value, int_fast16_t exp)
	{
		if (round_bit(frac_bits, value, exp))
		{
			if (sticky_bit(frac_bits, value, exp)) // > ulp/2 away
				return true;

			// = ulp/2 away, round towards even result, decided by LSB of mantissa
			if (exp <= 0) // denormal
				return denorm_frac(frac_bits, value, exp) & 0x1;
			return frac(frac_bits, value) & 0x1;
		}
		return false; // < ulp/2 away
	}

	inline uint_t denorm_pack(uint_fast8_t frac_bits, bool sign, uint_t frac)
	{
		return PACK(sign, 0, frac << (NUM_BITS_FRAC - frac_bits));
	}

	inline uint_t pack(uint_fast8_t exp_bits, uint_fast8_t frac_bits, bool sign, int_fast16_t exp, uint_t frac)
	{
		auto const bias = transprecision_floating_point::bias(exp_bits);
		auto const inf_exp = transprecision_floating_point::inf_exp(exp_bits);

		if (exp == inf_exp)   // Inf or NaN
			exp = INF_EXP;
		else
			exp = (exp - bias) + BIAS;

		return PACK(sign, exp, frac << (NUM_BITS_FRAC - frac_bits));
	}

	inline int_t rounding_value(uint_fast8_t exp_bits, uint_fast8_t frac_bits, float value, int_fast16_t exp, bool sign)
	{
		if (EXPONENT(CAST_TO_INT(value)) == 0) // Denorm backend format
			return denorm_pack(frac_bits, sign, 0x1);
		if (exp <= 0) // Denorm target format
			return pack(exp_bits, frac_bits, sign, -frac_bits + 1, 0);
		return pack(exp_bits, frac_bits, sign, exp - frac_bits, 0);
	}

	inline bool inf_rounding(uint_fast8_t exp_bits, uint_fast8_t frac_bits, float value, int_fast16_t exp, bool sign, bool plus)
	{
		if (round_bit(frac_bits, value, exp) || sticky_bit(frac_bits, value, exp))
			return (plus ^ sign);
		return false;
	}

	template<typename T>
	T sanitize(uint_fast8_t exp_bits, uint_fast8_t frac_bits, T value)
	{
		auto value_ = float(value);

		// Sign
		auto sign = transprecision_floating_point::sign(value_);

		// Exponent
		auto exp = exponent(exp_bits, value_);


		if (!(exp == INF_EXP))
		{
			// Rounding mode
#ifdef  __CUDA_ARCH__
			if (nearest_rounding(frac_bits, value_, exp))
			{
				int_t rounding_value = transprecision_floating_point::rounding_value(exp_bits, frac_bits, value_, exp, sign);
				value_ += CAST_TO_FP(rounding_value);
	}

			__threadfence_block();
#else
			auto const mode = fegetround();
			if (mode == FE_TONEAREST && nearest_rounding(frac_bits, value_, exp))
			{
				int_t rounding_value = transprecision_floating_point::rounding_value(exp_bits, frac_bits, value_, exp, sign);
				value_ += CAST_TO_FP(rounding_value);
			}
			else if (mode == FE_UPWARD && inf_rounding(exp_bits, frac_bits, value_, exp, sign, true))
			{
				int_t rounding_value = transprecision_floating_point::rounding_value(exp_bits, frac_bits, value_, exp, sign);
				value_ += CAST_TO_FP(rounding_value);
			}
			else if (mode == FE_DOWNWARD && inf_rounding(exp_bits, frac_bits, value_, exp, sign, false))
			{
				int_t rounding_value = transprecision_floating_point::rounding_value(exp_bits, frac_bits, value_, exp, sign);
				value_ += CAST_TO_FP(rounding_value);
			}

			_ReadWriteBarrier();
#endif

			// Recompute exponent value after rounding
			exp = exponent(exp_bits, value_);
}
#ifdef  __CUDA_ARCH__
#else
#endif
		// Exponent of NaN and Inf (target format)
		auto const inf_exp = transprecision_floating_point::inf_exp(exp_bits);

		// Mantissa
		auto frac = transprecision_floating_point::frac(frac_bits, value_);

		if (EXPONENT(CAST_TO_INT(value_)) == 0) // Denorm backend format - represented format also denormal
		{
			CAST_TO_INT(value_) = denorm_pack(frac_bits, sign, frac);
			return T(value_);
		}

		if (exp <= 0) // Denormalized value in the target format (saved in normalized format in the backend value)
		{
			uint_t const denorm = denorm_frac(frac_bits, value_, exp);
			if (denorm == 0) // value too low to be represented, return zero
			{
				CAST_TO_INT(value_) = PACK(sign, 0, 0);
				return T(value_);
			}
			if (frac_bits < NUM_BITS_FRAC) // Remove additional precision
			{
				auto shift = -exp + 1;
				if (shift < NUM_BITS_FRAC)
				{
					frac >>= shift;
					frac <<= shift;
				}
				else
					frac = 0;
			}
		}
		else if (exp == INF_EXP && (CAST_TO_INT(value_) & MASK_FRAC)) // NaN
		{
			exp = inf_exp;
		}
		else if (exp == INF_EXP) // Inf
		{
			exp = inf_exp;
		}
		else if (exp >= inf_exp) // Out of bounds for target format: set infinity
		{
			exp = inf_exp;
			frac = 0UL;
		}

		CAST_TO_INT(value_) = pack(exp_bits, frac_bits, sign, exp, frac);

		return T(value_);
	}

	template<typename T>
	__global__ void _sanitize_kernel(T* data, size_t n, size_t m, uint_fast8_t exp_bits, uint_fast8_t frac_bits)
	{
		size_t const i = blockIdx.x*blockDim.x + threadIdx.x;
		size_t const j = blockIdx.y*blockDim.y + threadIdx.y;

		if (i < n && j < m)
			data[i * m + j] = sanitize(exp_bits, frac_bits, data[i * m + j]);
	}

	template<typename T>
	void _sanitize(tensor_data<T> const& data, tensor_shape const& shape, tensor_precision const& precision)
	{
		auto tensor_dim = _calculate_block_size(shape);
		_sanitize_kernel << <tensor_dim.blocks_per_grid, tensor_dim.threads_per_block >> > (data.get(), tensor_dim.n, tensor_dim.m, precision.exp_bits, precision.frac_bits);
		CHECK_CUDA_ERROR(cudaGetLastError());
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */
}