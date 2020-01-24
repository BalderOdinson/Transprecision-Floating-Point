#pragma once
#include "../tensor/tensor_lib.h"
#include "../tensor/tensor_extensions.h"
#include "../tensor/convolution_extensions.h"

namespace transprecision_floating_point
{
	template<typename Weights, typename Bias, typename WeightGrads, typename BiasGrads>
	struct backward_grads
	{
		backward_grads(tensor<Weights>* weights, tensor<WeightGrads>&& grad_weights, tensor<Bias>* bias, tensor<BiasGrads>&& grad_bias);
		tensor<Weights>& weights() { return *weights_; };
		tensor<WeightGrads>& grad_weights() { return grad_weights_; };
		tensor<Bias>& bias() { return *bias_; };
		tensor<BiasGrads>& grad_bias() { return grad_bias_; };

	private:
		tensor<Weights>* weights_;
		tensor<WeightGrads> grad_weights_;
		tensor<Bias>* bias_;
		tensor<BiasGrads> grad_bias_;
	};

	template <typename Weights, typename Bias, typename WeightGrads, typename BiasGrads>
	backward_grads<Weights, Bias, WeightGrads, BiasGrads>::backward_grads(tensor<Weights>* weights,
		tensor<WeightGrads>&& grad_weights, tensor<Bias>* bias, tensor<BiasGrads>&& grad_bias) :
		weights_(weights),
		grad_weights_(std::move(grad_weights)),
		bias_(bias),
		grad_bias_(std::move(grad_bias))
	{
	}

	struct layer
	{
		virtual ~layer() = default;
		virtual tensor<float> forward(tensor<float>&& inputs) = 0;
		virtual tensor<float> forward_inputs(tensor<float>&& inputs) = 0; 
		virtual tensor<float> backward_inputs(tensor<float>&& grads) = 0;
		virtual backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) = 0;
		bool has_params;
		std::string name;
	};
}
