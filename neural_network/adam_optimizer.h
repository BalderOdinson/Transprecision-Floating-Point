#pragma once
#include "deep_model.h"

namespace transprecision_floating_point
{
	template<typename T>
	struct adam_optimizer
	{
		explicit adam_optimizer(T learning_rate, T decay_s, T decay_r, T delta);
		template<typename Loss>
		tensor<T> train_epoch(deep_model<Loss>& model, tensor<T> const& x, tensor<T> const& y_oh);

	private:
		T learning_rate_;
		T decay_s_;
		T decay_r_;
		T delta_;
		tensor<T> weight_s_;
		tensor<T> weight_r_;
		tensor<T> bias_s_;
		tensor<T> bias_r_;
		size_t t_{ 0 };
	};

	template <typename T>
	adam_optimizer<T>::adam_optimizer(
		T learning_rate,
		T decay_s,
		T decay_r,
		T delta) :
		learning_rate_(learning_rate),
		decay_s_(decay_s),
		decay_r_(decay_r),
		delta_(delta)
	{
	}

	template <typename T>
	template <typename Loss>
	tensor<T> adam_optimizer<T>::train_epoch(deep_model<Loss>& model, tensor<T> const& x,
		tensor<T> const& y_oh)
	{
		auto logits = model.forward_pass(x);
		auto grads = model.backward_pass(logits, y_oh);

		++t_;

		for (size_t i = 0; i < grads.size(); ++i)
		{
			auto grad_weights = grads[i].grad_weights();
			auto grad_bias = grads[i].grad_bias();
			if (weight_s_.shape() != grad_weights.shape())
			{
				weight_s_ = tensor<T>(grad_weights.shape(), T(0));
				weight_r_ = tensor<T>(grad_weights.shape(), T(0));
				bias_s_ = tensor<T>(grad_bias.shape(), T(0));
				bias_r_ = tensor<T>(grad_bias.shape(), T(0));
			}

			dnn_extensions::update_weights_adam(
				grads[i].weights(), grads[i].bias(), 
				grad_weights, grad_bias, 
				weight_s_, bias_s_, weight_r_, bias_r_, 
				learning_rate_, decay_s_, decay_r_, t_, delta_);
		}

		return logits;
	}
}