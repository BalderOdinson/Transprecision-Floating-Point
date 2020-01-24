#pragma once
#include <utility>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <memory>
#include <cmath>

#include <string>
#include <iostream>
#include <unordered_map>
#include <functional>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <device_launch_parameters.h>
#include "tensor_types.h"
#include "tensor_cuda_functions.h"
#include <random>

namespace transprecision_floating_point
{
	namespace cuda_blas
	{
		std::string to_string(tensor_shape shape);

		template<typename Type>
		struct tensor
		{
			tensor() = default;
			tensor(tensor<Type> const& other);
			tensor(tensor<Type>&& other) noexcept;
			explicit tensor(tensor_shape shape, Type default_element = 0);
			~tensor() = default;

			template<class Other>
			friend struct tensor;

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
			tensor<Type>& minimum(Type min);
			template<typename Function>
			tensor<Type> apply(Function func);

			template<typename Type, typename Function>
			friend tensor<Type> produce(tensor<Type> lhs, tensor<Type> const& rhs, Function function);

			std::vector<tensor<Type>> split(size_t count);

			template<typename Type>
			friend std::pair<tensor<Type>, tensor<Type>> permutate(tensor<Type> const& first, tensor<Type> const& second, size_t seed);

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
			tensor_data<Type> data_{ nullptr };
			Type default_element_{ 0 };
		};


#include "tensor_base_class_functions.h"
#include "tensor_operators.h"
#include "tensor_arithmetic_operations.h"
		template <typename Type>
		std::pair<tensor<Type>, tensor<Type>> permutate(tensor<Type> const& first, tensor<Type> const& second, size_t seed)
		{
			if (first.shape_.first != second.shape_.first)
				throw std::runtime_error("Invalid data shape!");

			std::uniform_int_distribution<size_t> dist;
			typedef std::uniform_int_distribution<size_t>::param_type param_t;
			std::mt19937 rng(seed);

			tensor<Type> first_permutated;
			first_permutated.shape_ = first.shape_;
			first_permutated.data_ = create_tensor_data<Type>(first.shape_.first * first.shape_.second);
			tensor<Type> second_permutated;
			second_permutated.shape_ = second.shape_;
			second_permutated.data_ = create_tensor_data<Type>(second.shape_.first * second.shape_.second);

			cuda_blas::_permutate(first.data_, second.data_, first_permutated.data_, second_permutated.data_, first.shape_, second.shape_);

			return { first_permutated, second_permutated };
		}
	}
}
