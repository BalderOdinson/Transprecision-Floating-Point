#pragma once
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct backward_grads
		{
			backward_grads(tensor<Type>* weights, tensor<Type>&& grad_weights, tensor<Type>* bias, tensor<Type>&& grad_bias);
			tensor<Type>& weights();
			tensor<Type>& grad_weights();
			tensor<Type>& bias();
			tensor<Type>& grad_bias();

		private:
			tensor<Type>* weights_;
			tensor<Type> grad_weights_;
			tensor<Type>* bias_;
			tensor<Type> grad_bias_;
		};

		template <typename Type>
		backward_grads<Type>::backward_grads(
			tensor<Type>* weights, tensor<Type>&& grad_weights,
			tensor<Type>* bias, tensor<Type>&& grad_bias) :
			weights_(weights),
			grad_weights_(std::move(grad_weights)),
			bias_(bias),
			grad_bias_(std::move(grad_bias))
		{
		}

		template <typename Type>
		tensor<Type>& backward_grads<Type>::weights()
		{
			return *weights_;
		}

		template <typename Type>
		tensor<Type>& backward_grads<Type>::grad_weights()
		{
			return grad_weights_;
		}

		template <typename Type>
		tensor<Type>& backward_grads<Type>::bias()
		{
			return *bias_;
		}

		template <typename Type>
		tensor<Type>& backward_grads<Type>::grad_bias()
		{
			return grad_bias_;
		}

		template<typename Type>
		struct layer
		{
			virtual ~layer() = default;
			virtual tensor<Type> forward(tensor<Type> const& inputs) = 0;
			virtual tensor<Type> backward_inputs(tensor<Type> const& grads) = 0;
			virtual backward_grads<Type> backward_params(tensor<Type> const& grads) = 0;
			bool has_params;
			std::string name;
		};
	}
}
