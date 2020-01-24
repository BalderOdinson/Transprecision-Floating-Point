#pragma once
#include "layer.h"
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct mean_squared_error
		{
			mean_squared_error() = default;
			Type forward(tensor<Type> const& x, tensor<Type> const& y);
			tensor<Type> backward_inputs(tensor<Type> const& x, tensor<Type> const& y);
			backward_grads<Type> backward_params() const;
			bool const has_params = false;
		};

		template <typename Type>
		Type mean_squared_error<Type>::forward(tensor<Type> const& x, tensor<Type> const& y)
		{
			auto const n = x.shape().first;

			return (y - x).apply([] DEVICE_FUNCTION(Type value) { return value * value; }).sum() / n;
		}

		template <typename Type>
		tensor<Type> mean_squared_error<Type>::backward_inputs(tensor<Type> const& x,
			tensor<Type> const& y)
		{
			auto const n = x.shape().first;
			return (x - y) / Type(n / 2.);
		}

		template <typename Type>
		backward_grads<Type> mean_squared_error<Type>::backward_params() const
		{
			throw std::runtime_error("Function not implemented!");
		}
	}
}
