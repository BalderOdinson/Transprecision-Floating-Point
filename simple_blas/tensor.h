#pragma once
#include <utility>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <memory>
#include <cmath>

#include "thread_safe_rand.h"
#include <string>
#include <iostream>
#include "../flexfloat/flexfloat.hpp"

namespace transprecision_floating_point::simple_blas
{
	using tensor_shape = std::pair<size_t, size_t>;
	using tensor_index = tensor_shape;
	template <typename Type, size_t Height, size_t Width>
	using array2d = std::array<std::array<Type, Width>, Height>;
	std::string to_string(tensor_shape shape);

	template<typename Type>
	struct tensor
	{
		tensor() = default;
		tensor(tensor<Type> const& other);
		tensor(tensor<Type>&& other) noexcept;
		explicit tensor(tensor_shape shape, Type default_element = 0);
		~tensor() = default;

		tensor<Type>& operator=(tensor<Type> const& other);
		tensor<Type>& operator=(tensor<Type>&& other) noexcept;

		template <typename T> 
		tensor<T> cast();

		Type& operator[](tensor_index idx);
		Type const& operator[](tensor_index idx) const;

		tensor<Type> operator+=(tensor<Type> const& other);
		tensor<Type> operator+=(Type other);
		tensor<Type> operator-=(tensor<Type> const& other);
		tensor<Type> operator-=(Type other);
		tensor<Type> operator*=(tensor<Type> const& other);
		tensor<Type> operator*=(Type other);
		tensor<Type> operator/=(tensor<Type> const& other);
		tensor<Type> operator/=(Type other);

		tensor<Type>& transpose();

		tensor<Type>& dot(tensor<Type> const& other);

		template<uint_fast8_t Axis>
		tensor<Type> sum() const;
		Type sum() const;
		template<uint_fast8_t Axis>
		tensor<Type> max() const;
		Type max() const;
		template<uint_fast8_t Axis>
		tensor<Type> min() const;
		Type min() const;
		template<uint_fast8_t Axis>
		tensor<size_t> argmax() const;
		tensor_index argmax() const;
		template<uint_fast8_t Axis>
		tensor<size_t> argmin() const;
		tensor_index argmin() const;
		template<typename Reduction, uint_fast8_t Axis>
		tensor<Type> reduce(Reduction reduction) const;
		template<typename Reduction>
		Type reduce(Reduction reduction) const;

		tensor<Type>& log();
		tensor<Type>& exp();
		tensor<Type>& abs();
		tensor<Type>& sqrt();
		tensor<Type>& maximum(Type max);
		tensor<Type>& minimum(Type max);
		template<typename Function>
		tensor<Type> apply(Function func);

		tensor_shape shape() const;

		template <size_t TensorHeight, size_t TensorWidth>
		static tensor create(std::initializer_list<Type> data);
		static tensor create(size_t tensor_height, size_t tensor_width, std::initializer_list<Type> data);
		template <size_t TensorHeight, size_t TensorWidth>
		static tensor create(array2d<Type, TensorHeight, TensorWidth> const& data);
		template <size_t TensorHeight, size_t TensorWidth, typename Distribution>
		static tensor create_random(Distribution distribution);
		template <typename Distribution>
		static tensor create_random(size_t tensor_height, size_t tensor_width, Distribution distribution);
		template <size_t TensorHeight, size_t TensorWidth>
		static tensor identity();
		static tensor identity(size_t tensor_height, size_t tensor_width);
		template <size_t TensorHeight, size_t TensorWidth>
		static tensor diagonal(std::array<Type, TensorHeight> const& data);
		static tensor diagonal(size_t tensor_height, size_t tensor_width, std::initializer_list<Type> data);

	private:
		tensor_shape shape_{ 0,0 };
		std::unique_ptr<Type[]> data_{ nullptr };
		Type default_element_{ 0 };
	};

#include "tensor_base_class_functions.h"
#include "tensor_operators.h"
#include "tensor_arithmetic_operations.h"
}
