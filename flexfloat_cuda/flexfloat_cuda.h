#pragma once
#include "cuda_runtime.h"
#include <cstdint>
#include <ostream>
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
	namespace cuda
	{
		// Get the first unused global index value in the  private storage of std::ios_base
		static int get_manipulator_id() {
			static auto id = std::ios_base::xalloc();
			return id;
		}


		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		struct flexfloat_cuda
		{
			__host__ __device__ flexfloat_cuda();
			template<typename U>
			__host__ __device__ flexfloat_cuda(U const& value);
			__host__ __device__ explicit flexfloat_cuda(flexfloat_cuda const& other);
			__host__ __device__ explicit flexfloat_cuda(flexfloat_cuda&& other) noexcept;
			template<uint_fast8_t ExpBitsOther, uint_fast8_t FracBitsOther>
			__host__ __device__ explicit flexfloat_cuda(flexfloat_cuda<ExpBitsOther, FracBitsOther> const& other);
			template<uint_fast8_t ExpBitsOther, uint_fast8_t FracBitsOther>
			__host__ __device__ explicit flexfloat_cuda(flexfloat_cuda<ExpBitsOther, FracBitsOther>&& other) noexcept;

			__host__ __device__ flexfloat_cuda& operator=(flexfloat_cuda const& other);
			__host__ __device__ flexfloat_cuda& operator=(flexfloat_cuda&& other) noexcept;

			__host__ __device__  float& value();
			__host__ __device__  float const& value() const;

			/*------------------------------------------------------------------------
			| OPERATOR OVERLOADS: CASTS
			*------------------------------------------------------------------------*/
			__host__ __device__ explicit operator float() const;
			__host__ __device__ explicit operator double() const;
			__host__ __device__ explicit operator long double() const;

			/*------------------------------------------------------------------------
		|	OPERATOR OVERLOADS: Arithmetic
			*------------------------------------------------------------------------*/
			__host__ __device__ flexfloat_cuda operator-() const;
			__host__ __device__ flexfloat_cuda operator+() const;
			__host__ __device__ friend flexfloat_cuda operator+(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				flexfloat_cuda lhs1(lhs);
				return lhs1 += rhs;
			}
			__host__ __device__ friend flexfloat_cuda operator-(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				flexfloat_cuda lhs1(lhs);
				return lhs1 -= rhs;
			}
			__host__ __device__ friend flexfloat_cuda operator*(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				flexfloat_cuda lhs1(lhs);
				return lhs1 *= rhs;
			}
			__host__ __device__ friend flexfloat_cuda operator/(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				flexfloat_cuda lhs1(lhs);
				return lhs1 /= rhs;
			}

			/*------------------------------------------------------------------------
			| OPERATOR OVERLOADS: Relational operators
			*------------------------------------------------------------------------*/
			__host__ __device__ friend bool operator==(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return lhs.value_ == rhs.value_;
			}
			__host__ __device__ friend bool operator!=(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return !(lhs == rhs); 
			}
			__host__ __device__ friend bool operator>(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return rhs < lhs; 
			}
			__host__ __device__ friend bool operator<(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return lhs.value_ < rhs.value_;
			}
			__host__ __device__ friend bool operator>=(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return !(lhs < rhs); 
			}
			__host__ __device__ friend bool operator<=(flexfloat_cuda const& lhs, flexfloat_cuda const& rhs)
			{
				return !(rhs < lhs); 
			}

			/*------------------------------------------------------------------------
			| OPERATOR OVERLOADS: Compound assignment operators (no bitwise ops)
			*------------------------------------------------------------------------*/
			__host__ __device__ flexfloat_cuda operator+=(flexfloat_cuda const& other);
			__host__ __device__ flexfloat_cuda operator-=(flexfloat_cuda const& other);
			__host__ __device__ flexfloat_cuda operator*=(flexfloat_cuda const& other);
			__host__ __device__ flexfloat_cuda operator/=(flexfloat_cuda const& other);

			/*------------------------------------------------------------------------
			| OPERATOR OVERLOADS: IO streams operators
			*------------------------------------------------------------------------*/
			friend std::ostream& operator<<(std::ostream& os, flexfloat_cuda const& obj)
			{
				if (os.iword(get_manipulator_id()) == 0)
				{
					os << float(obj);
				}
				else
				{
					int_fast16_t exp = obj.exponent();
					uint_t frac;
					if (exp <= 0) {
						frac = flexfloat_cuda::denorm_frac(obj, exp);
						exp = 0;
					}
					else
						frac = flexfloat_cuda::frac(obj);

					os << obj.sign() << "-";
					os << std::bitset<ExpBits>(exp) << "-";
					os << std::bitset<FracBits>(frac);
				}
				return os;
			}

		protected:
			__host__ __device__ void sanitize();

		private:
			__host__ __device__ bool sign() const;
			__host__ __device__ int_fast16_t exponent() const;
			__host__ __device__ static int_fast16_t bias();
			__host__ __device__ static int_fast16_t inf_exp();
			__host__ __device__ static bool round_bit(flexfloat_cuda const& a, int_fast16_t exp);
			__host__ __device__ static bool sticky_bit(flexfloat_cuda const& a, int_fast16_t exp);
			__host__ __device__ static uint_t denorm_frac(flexfloat_cuda const& a, int_fast16_t exp);
			__host__ __device__ static uint_t frac(flexfloat_cuda const& a);
			__host__ __device__ static bool nearest_rounding(flexfloat_cuda const& a, int_fast16_t exp);
			__host__ __device__ static uint_t denorm_pack(bool sign, uint_t frac);
			__host__ __device__ static uint_t pack(bool sign, int_fast16_t exp, uint_t frac);
			__host__ __device__ static int_t rounding_value(flexfloat_cuda const& a, int_fast16_t exp, bool sign);
			__host__ __device__ static bool inf_rounding(flexfloat_cuda const& a, int_fast16_t exp, bool sign, bool plus);

			float value_;
		};

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda() : value_(0.0f)
		{
			sanitize();
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		template <typename U>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda(U const& value) : value_(value)
		{
			sanitize();
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda(flexfloat_cuda const& other) : value_(other.value_)
		{
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda(flexfloat_cuda&& other) noexcept : value_(other.value_)
		{
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		template <uint_fast8_t ExpBitsOther, uint_fast8_t FracBitsOther>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda(flexfloat_cuda<ExpBitsOther, FracBitsOther> const& other) : value_(other.value())
		{
			sanitize();
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		template <uint_fast8_t ExpBitsOther, uint_fast8_t FracBitsOther>
		flexfloat_cuda<ExpBits, FracBits>::flexfloat_cuda(flexfloat_cuda<ExpBitsOther, FracBitsOther>&& other) noexcept : value_(other.value())
		{
			sanitize();
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>& flexfloat_cuda<ExpBits, FracBits>::operator=(flexfloat_cuda const& other)
		{
			value_ = other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>& flexfloat_cuda<ExpBits, FracBits>::operator=(flexfloat_cuda&& other) noexcept
		{
			value_ = other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		float& flexfloat_cuda<ExpBits, FracBits>::value()
		{
			return value_;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		float const& flexfloat_cuda<ExpBits, FracBits>::value() const
		{
			return value_;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::operator float() const
		{
			return value_;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::operator double() const
		{
			return value_;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits>::operator long double() const
		{
			return value_;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator-() const
		{
			return flexfloat_cuda(-value_);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator+() const
		{
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator+=(flexfloat_cuda const& other)
		{
			value_ += other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator-=(flexfloat_cuda const& other)
		{
			value_ -= other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator*=(flexfloat_cuda const& other)
		{
			value_ *= other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		flexfloat_cuda<ExpBits, FracBits> flexfloat_cuda<ExpBits, FracBits>::operator/=(flexfloat_cuda const& other)
		{
			value_ /= other.value_;
			sanitize();
			return *this;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		void flexfloat_cuda<ExpBits, FracBits>::sanitize()
		{
			// This case does not require to be sanitized
			if (ExpBits == NUM_BITS_EXP && ExpBits == NUM_BITS_FRAC)
				return;

			// Sign
			auto sign = this->sign();

			// Exponent
			auto exp = this->exponent();


			if (!(exp == INF_EXP || FracBits == NUM_BITS_FRAC))
			{
				// Rounding mode
#ifdef  __CUDA_ARCH__
				if(flexfloat_cuda::nearest_rounding(*this, exp))
				{
					int_t rounding_value = flexfloat_cuda::rounding_value(*this, exp, sign);
					value_ += CAST_TO_FP(rounding_value);	
				}

				__threadfence_block();
#else
				auto const mode = fegetround();
				if (mode == FE_TONEAREST && flexfloat_cuda::nearest_rounding(*this, exp))
				{
					int_t rounding_value = flexfloat_cuda::rounding_value(*this, exp, sign);
					value_ += CAST_TO_FP(rounding_value);
				}
				else if (mode == FE_UPWARD && flexfloat_cuda::inf_rounding(*this, exp, sign, true))
				{
					int_t rounding_value = flexfloat_cuda::rounding_value(*this, exp, sign);
					value_ += CAST_TO_FP(rounding_value);
				}
				else if (mode == FE_DOWNWARD && flexfloat_cuda::inf_rounding(*this, exp, sign, false))
				{
					int_t rounding_value = flexfloat_cuda::rounding_value(*this, exp, sign);
					value_ += CAST_TO_FP(rounding_value);
				}

				_ReadWriteBarrier();
#endif

				// Recompute exponent value after rounding
				exp = exponent();
			}
#ifdef  __CUDA_ARCH__
#else
#endif
			// Exponent of NaN and Inf (target format)
			auto const inf_exp = flexfloat_cuda::inf_exp();

			// Mantissa
			auto frac = flexfloat_cuda::frac(*this);

			if (EXPONENT(CAST_TO_INT(value_)) == 0) // Denorm backend format - represented format also denormal
			{
				CAST_TO_INT(value_) = flexfloat_cuda::denorm_pack(sign, frac);
				return;
			}

			if (exp <= 0) // Denormalized value in the target format (saved in normalized format in the backend value)
			{
				uint_t const denorm = flexfloat_cuda::denorm_frac(*this, exp);
				if (denorm == 0) // value too low to be represented, return zero
				{
					CAST_TO_INT(value_) = PACK(sign, 0, 0);
					return;
				}
				if (FracBits < NUM_BITS_FRAC) // Remove additional precision
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

			CAST_TO_INT(value_) = flexfloat_cuda::pack(sign, exp, frac);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		bool flexfloat_cuda<ExpBits, FracBits>::sign() const
		{
			return (CAST_TO_INT(value_)) >> (NUM_BITS - 1);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		int_fast16_t flexfloat_cuda<ExpBits, FracBits>::exponent() const
		{
			auto const a_exp = EXPONENT(CAST_TO_INT(value_));

			auto const bias = flexfloat_cuda::bias();

			if (a_exp == 0 || a_exp == INF_EXP)
				return a_exp;

			return (a_exp - BIAS) + bias;
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		int_fast16_t flexfloat_cuda<ExpBits, FracBits>::bias()
		{
			return int_fast16_t((int_fast16_t(1) << (ExpBits - 1)) - 1);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		int_fast16_t flexfloat_cuda<ExpBits, FracBits>::inf_exp()
		{
			return int_fast16_t((int_fast16_t(1) << ExpBits) - 1);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		bool flexfloat_cuda<ExpBits, FracBits>::round_bit(flexfloat_cuda const& a, int_fast16_t exp)
		{
			if (exp <= 0 && EXPONENT(CAST_TO_INT(a.value_)) != 0)
			{
				auto const shift = (-exp + 1);
				uint_t denorm = 0;
				if (shift < NUM_BITS)
					denorm = ((CAST_TO_INT(a.value_) & MASK_FRAC | MASK_FRAC_MSB)) >> shift;
				return denorm & (UINT_C(0x1) << (NUM_BITS_FRAC - FracBits - 1));
			}
			return CAST_TO_INT(a.value_) & (UINT_C(0x1) << (NUM_BITS_FRAC - FracBits - 1));
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		bool flexfloat_cuda<ExpBits, FracBits>::sticky_bit(flexfloat_cuda const& a, int_fast16_t exp)
		{
			if (exp <= 0 && EXPONENT(CAST_TO_INT(a.value_)) != 0)
			{
				auto const shift = (-exp + 1);
				uint_t denorm = 0;
				if (shift < NUM_BITS)
					denorm = ((CAST_TO_INT(a.value_) & MASK_FRAC) | MASK_FRAC_MSB) >> shift;
				return (denorm & (MASK_FRAC >> (FracBits + 1))) ||
					(((denorm & MASK_FRAC) == 0) && (CAST_TO_INT(a.value_) != 0));
			}
			return CAST_TO_INT(a.value_) & (MASK_FRAC >> (FracBits + 1));
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		uint_t flexfloat_cuda<ExpBits, FracBits>::denorm_frac(flexfloat_cuda const& a, int_fast16_t exp)
		{
			if (EXPONENT(CAST_TO_INT(a.value_)) == 0) // Denormalized backend value
				return (CAST_TO_INT(a.value_) & MASK_FRAC) >> (NUM_BITS_FRAC - FracBits);

			// Denormalized target value (in normalized backend value)
			unsigned short const shift = NUM_BITS_FRAC - FracBits - exp + 1;
			if (shift >= NUM_BITS) return false;
			return (((CAST_TO_INT(a.value_) & MASK_FRAC) | MASK_FRAC_MSB) >> shift);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		uint_t flexfloat_cuda<ExpBits, FracBits>::frac(flexfloat_cuda const& a)
		{
			return (CAST_TO_INT(a.value_) & MASK_FRAC) >> (NUM_BITS_FRAC - FracBits);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		bool flexfloat_cuda<ExpBits, FracBits>::nearest_rounding(flexfloat_cuda const& a, int_fast16_t exp)
		{
			if (flexfloat_cuda::round_bit(a, exp))
			{
				if (flexfloat_cuda::sticky_bit(a, exp)) // > ulp/2 away
					return true;

				// = ulp/2 away, round towards even result, decided by LSB of mantissa
				if (exp <= 0) // denormal
					return flexfloat_cuda::denorm_frac(a, exp) & 0x1;
				return flexfloat_cuda::frac(a) & 0x1;
			}
			return false; // < ulp/2 away
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		uint_t flexfloat_cuda<ExpBits, FracBits>::denorm_pack(bool sign, uint_t frac)
		{
			return PACK(sign, 0, frac << (NUM_BITS_FRAC - FracBits));
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		uint_t flexfloat_cuda<ExpBits, FracBits>::pack(bool sign, int_fast16_t exp, uint_t frac)
		{
			auto const bias = flexfloat_cuda::bias();
			auto const inf_exp = flexfloat_cuda::inf_exp();

			if (exp == inf_exp)   // Inf or NaN
				exp = INF_EXP;
			else
				exp = (exp - bias) + BIAS;

			return PACK(sign, exp, frac << (NUM_BITS_FRAC - FracBits));
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		int_t flexfloat_cuda<ExpBits, FracBits>::rounding_value(flexfloat_cuda const& a, int_fast16_t exp, bool sign)
		{
			if (EXPONENT(CAST_TO_INT(a.value_)) == 0) // Denorm backend format
				return denorm_pack(sign, 0x1);
			if (exp <= 0) // Denorm target format
				return pack(sign, -FracBits + 1, 0);
			return pack(sign, exp - FracBits, 0);
		}

		template <uint_fast8_t ExpBits, uint_fast8_t FracBits>
		bool flexfloat_cuda<ExpBits, FracBits>::inf_rounding(flexfloat_cuda const& a, int_fast16_t exp, bool sign,
		                                                     bool plus)
		{
			if (flexfloat_cuda::round_bit(a, exp) || flexfloat_cuda::sticky_bit(a, exp))
				return (plus ^ sign);
			return false;
		}
		}
		}

