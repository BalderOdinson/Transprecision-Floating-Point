#pragma once
#include "deep_model.h"

namespace transprecision_floating_point
{
	struct sgd_optimizer
	{
		explicit sgd_optimizer(float learning_rate);
		template<typename Loss>
		tensor<float> train_epoch(deep_model<Loss>& model, tensor<float> const& x, tensor<float> const& y_oh);

	private:
		float learning_rate_;
	};

	inline sgd_optimizer::sgd_optimizer(float learning_rate) : learning_rate_(learning_rate)
	{
	}

	template <typename Loss>
	tensor<float> sgd_optimizer::train_epoch(deep_model<Loss>& model, tensor<float> const& x,
		tensor<float> const& y_oh)
	{
		auto logits = model.forward_pass(x);
		auto grads = model.backward_pass(logits, y_oh);

		for (size_t i = 0; i < grads.size(); ++i)
		{
			tensor_extensions::geam(1.f, grads[i].weights(), false, -learning_rate_, grads[i].grad_weights(), false, grads[i].weights());
			tensor_extensions::geam(1.f, grads[i].bias(), false, -learning_rate_, grads[i].grad_bias(), false, grads[i].bias());
		}

		return logits;
	}
}