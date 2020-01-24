#pragma once
#include "layer.h"
#include <fstream>
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{

		template<typename Type>
		struct fully_connected_layer : layer<Type>
		{
			template<typename WeightsInitializer, typename BiasInitializer>
			fully_connected_layer(size_t input_layer, size_t output_count, WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name);
			tensor<Type> forward(tensor<Type> const& inputs) override;
			tensor<Type> backward_inputs(tensor<Type> const& grads) override;
			backward_grads<Type> backward_params(tensor<Type> const& grads) override;

			tensor<Type> weights_;
			tensor<Type> bias_;
		private:
			tensor_shape shape_;
			tensor<Type> x_;
		};

		template <typename Type>
		template <typename WeightsInitializer, typename BiasInitializer>
		fully_connected_layer<Type>::fully_connected_layer(size_t input_layer, size_t output_count,
			WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name)
		{
			//std::fstream file(name, std::fstream::out);
			layer<Type>::has_params = true;
			layer<Type>::name = std::move(name);
			shape_ = { input_layer, output_count };
			weights_ = weights_initializer({ output_count, input_layer });
			//file << weights_ << std::endl << std::endl;
			bias_ = bias_initializer({ 1, output_count });
		}

		template <typename Type>
		tensor<Type> fully_connected_layer<Type>::forward(tensor<Type> const& inputs)
		{
			x_ = inputs;
			return dot(inputs, transpose(weights_)) + bias_;
		}

		template <typename Type>
		tensor<Type> fully_connected_layer<Type>::backward_inputs(tensor<Type> const& grads)
		{
			return dot(grads, weights_);
		}

		template <typename Type>
		backward_grads<Type> fully_connected_layer<Type>::backward_params(tensor<Type> const& grads)
		{
			auto grad_weights = dot(transpose(grads), x_);
			auto grad_bias = grads.template sum<0>();

			return { &weights_, std::move(grad_weights), &bias_, std::move(grad_bias) };
		}
	}
}
