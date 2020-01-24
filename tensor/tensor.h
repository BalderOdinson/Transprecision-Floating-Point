#pragma once

#include "tensor_types.h"
#include <sstream>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "number_operators.h"
#include "tensor_lib_init.h"

namespace transprecision_floating_point
{
	template<typename T>
	struct tensor
	{
		template<class U>
		friend struct tensor;
		friend struct tensor_extensions;
		friend struct convolution_extensions;
		friend struct dnn_extensions;

		tensor();
		tensor(tensor const& other);
		tensor(tensor const& other, tensor_precision precision);
		tensor(tensor&& other) noexcept;
		tensor(tensor&& other, tensor_precision precision) noexcept;
		template<typename U>
		explicit tensor(tensor<U> const& other);
		template<typename U>
		tensor(tensor<U> const& other, tensor_precision precision);
		explicit tensor(tensor_shape const& shape);
		explicit tensor(tensor_shape const& shape, tensor_precision precision);
		explicit tensor(tensor_shape const& shape, T default_element);
		explicit tensor(tensor_shape const& shape, tensor_precision precision, T default_element);
		~tensor() = default;

		tensor& operator=(tensor const& other);
		tensor& operator=(tensor&& other) noexcept;
		template<typename U>
		tensor& operator=(tensor<U> const& other);

		void set_precision(tensor_precision precision);
		tensor_precision get_precision() const;

		T& operator[](tensor_index const& idx);
		T const& operator[](tensor_index const& idx) const;

		tensor& operator+=(tensor const& other);
		template<typename U> 
		tensor& operator+=(tensor<U> const& other);
		template<typename U>
		tensor& operator+=(U other);
		tensor& operator-=(tensor const& other);
		template<typename U>
		tensor& operator-=(tensor<U> const& other);
		template<typename U>
		tensor& operator-=(U other);
		tensor& operator*=(tensor const& other);
		template<typename U>
		tensor& operator*=(tensor<U> const& other);
		template<typename U>
		tensor& operator*=(U other);
		tensor& operator/=(tensor const& other);
		template<typename U>
		tensor& operator/=(tensor<U> const& other);
		template<typename U>
		tensor& operator/=(U other);

		tensor operator~() const;

		template<typename U>
		tensor dot(tensor<U> const& other) const;

		tensor sum(uint_fast8_t axis, bool keep_dims = true) const;
		T sum() const;
		tensor max(uint_fast8_t axis, bool keep_dims = true) const;
		T max() const;
		tensor min(uint_fast8_t axis, bool keep_dims = true) const;
		T min() const;
		tensor<size_t> argmax(uint_fast8_t axis, bool keep_dims = true) const;
		tensor_index argmax() const;
		tensor<size_t> argmin(uint_fast8_t axis, bool keep_dims = true) const;
		tensor_index argmin() const;
		template<typename Reduction>
		tensor reduce(Reduction reduction, uint_fast8_t axis, bool keep_dims = true) const;
		template<typename Reduction>
		T reduce(Reduction reduction) const;

		tensor& log();
		tensor& exp();
		tensor& abs();
		tensor& sqrt();
		tensor& maximum(T max);
		tensor& minimum(T min);
		template<typename Function>
		tensor apply(Function func);

		template <typename P, typename U, typename R, typename Function>
		friend tensor<R> produce(tensor<P> const& lhs, tensor<U> const& rhs, Function function);
		template<typename P, typename U, typename R, typename Function>
		friend tensor<R> produce(tensor<P> const& lhs, tensor<U> const& rhs, Function function, tensor_precision const& precision);

		void reshape(tensor_shape const& shape);

		tensor_shape const& shape() const;
		T* data() const;

		static tensor create(tensor_shape const& shape, std::initializer_list<T> data);
		static tensor create(tensor_shape const& shape, std::initializer_list<T> data, tensor_precision const& precision);
		template <typename Distribution>
		static tensor random(tensor_shape const& shape, Distribution distribution);
		template <typename Distribution>
		static tensor random(tensor_shape const& shape, Distribution distribution, tensor_precision const& precision);

	protected:
		void update_values();
		bool should_sanitize() const;

	private:
		tensor_data<T> data_;
		tensor_shape shape_;
		tensor_precision precision_;
	};

	template<typename T>
	tensor<T> operator+(tensor<T> lhs, tensor<T> const& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator+(tensor<T> lhs, U rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator+(U lhs, tensor<T> rhs)
	{
		rhs += lhs;
		return rhs;
	}

	template<typename T>
	tensor<T> operator-(tensor<T> lhs, tensor<T> const& rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator-(tensor<T> lhs, U rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator-(U lhs, tensor<T> rhs)
	{
		rhs -= lhs;
		return rhs;
	}

	template<typename T>
	tensor<T> operator*(tensor<T> lhs, tensor<T> const& rhs)
	{
		lhs *= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator*(tensor<T> lhs, U rhs)
	{
		lhs *= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator*(U lhs, tensor<T> rhs)
	{
		rhs *= lhs;
		return rhs;
	}

	template<typename T>
	tensor<T> operator/(tensor<T> lhs, tensor<T> const& rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator/(tensor<T> lhs, U rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	template<typename T, typename U>
	tensor<T> operator/(U lhs, tensor<T> rhs)
	{
		rhs /= lhs;
		return rhs;
	}

	template<typename T>
	std::ostream& operator<<(std::ostream& os, tensor<T> const& obj);

	std::string to_string(tensor_shape const& shape);
}

