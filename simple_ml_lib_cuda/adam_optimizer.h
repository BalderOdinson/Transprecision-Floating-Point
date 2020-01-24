#pragma once
#include "types.h"
#include "deep_model.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{

		template<typename Type>
		struct adam_optimizer
		{
			explicit adam_optimizer(Type learning_rate, Type decay_s, Type decay_r, Type delta);
			template<typename Loss>
			tensor<Type> train_epoch(deep_model<Type, Loss>& model, tensor<Type> const& x, tensor<Type> const& y_oh);

		private:
			Type learning_rate_;
			Type decay_s_;
			Type decay_r_;
			Type delta_;
			tensor<Type> weight_s_;
			tensor<Type> weight_r_;
			tensor<Type> bias_s_;
			tensor<Type> bias_r_;
			size_t t_{ 0 };
		};

		template <typename Type>
		adam_optimizer<Type>::adam_optimizer(
			Type learning_rate,
			Type decay_s,
			Type decay_r,
			Type delta) :
			learning_rate_(learning_rate),
			decay_s_(decay_s),
			decay_r_(decay_r),
			delta_(delta)
		{
		}

		template <typename Type>
		template <typename Loss>
		tensor<Type> adam_optimizer<Type>::train_epoch(deep_model<Type, Loss>& model, tensor<Type> const& x,
			tensor<Type> const& y_oh)
		{
			auto logits = model.feed_forward(x);
			auto grads = model.backward_pass(logits, y_oh);

			++t_;

			for (size_t i = 0; i < grads.size(); ++i)
			{
				auto grad_weights = grads[i].grad_weights();
				auto grad_bias = grads[i].grad_bias();
				if (weight_s_.shape() != grad_weights.shape())
				{
					weight_s_ = tensor<Type>(grad_weights.shape());
					weight_r_ = tensor<Type>(grad_weights.shape());
					bias_s_ = tensor<Type>(grad_bias.shape());
					bias_r_ = tensor<Type>(grad_bias.shape());
				}
				weight_s_ = decay_s_ * weight_s_ + (1 - decay_s_) * grad_weights;
				bias_s_ = decay_s_ * bias_s_ + (1 - decay_s_) * grad_bias;
				weight_r_ = decay_r_ * weight_r_ + (1 - decay_r_) * grad_weights * grad_weights;
				bias_r_ = decay_r_ * bias_r_ + (1 - decay_r_) * grad_bias * grad_bias;

				weight_s_ /= 1 - Type(powf(static_cast<float>(decay_s_), t_));
				bias_s_ /= 1 - Type(powf(static_cast<float>(decay_s_), t_));
				weight_r_ /= 1 - Type(powf(static_cast<float>(decay_r_), t_));
				bias_r_ /= 1 - Type(powf(static_cast<float>(decay_r_), t_));

				auto delta_weight = -learning_rate_ * (weight_s_ / (sqrt(weight_r_) + delta_));
				auto delta_bias = -learning_rate_ * (bias_s_ / (sqrt(bias_r_) + delta_));

				grads[i].weights() += delta_weight;
				grads[i].bias() += delta_bias;
			}

			return logits;
		}
	}
}