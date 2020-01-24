#pragma once
#include "layer.h"
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct relu_layer : layer<Type>
		{
			explicit relu_layer(std::string name);
			tensor<Type> forward(tensor<Type> const& inputs) override;
			tensor<Type> backward_inputs(tensor<Type> const& grads) override;
			backward_grads<Type> backward_params(tensor<Type> const& grads) override;

		private:
			tensor<Type> scores_;
		};

		template <typename Type>
		relu_layer<Type>::relu_layer(std::string name)
		{
			layer<Type>::has_params = false;
			layer<Type>::name = std::move(name);
		}

		template <typename Type>
		tensor<Type> relu_layer<Type>::forward(tensor<Type> const& inputs)
		{
			scores_ = maximum(inputs, Type(0));
			return scores_;
		}

		template <typename Type>
		tensor<Type> relu_layer<Type>::backward_inputs(tensor<Type> const& grads)
		{
			return produce(grads, scores_, [] DEVICE_FUNCTION(Type first, Type second) { return second > 0 ? first : 0; });
		}

		template <typename Type>
		backward_grads<Type> relu_layer<Type>::backward_params(tensor<Type> const& grads)
		{
			throw std::runtime_error("Function not implemented!");
		}
	}
}