#pragma once
#include "deep_model.h"
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct sgd_optimizer
		{
			explicit sgd_optimizer(Type learning_rate);
			template<typename Loss>
			tensor<Type> train_epoch(deep_model<Type, Loss>& model, tensor<Type> const& x, tensor<Type> const& y_oh);

		private:
			Type learning_rate_;
		};

		template <typename Type>
		sgd_optimizer<Type>::sgd_optimizer(Type learning_rate) : learning_rate_(learning_rate)
		{
		}

		template <typename Type>
		template <typename Loss>
		tensor<Type> sgd_optimizer<Type>::train_epoch(deep_model<Type, Loss>& model, tensor<Type> const& x,
			tensor<Type> const& y_oh)
		{
			auto logits = model.feed_forward(x);
			auto grads = model.backward_pass(logits, y_oh);

			for (size_t i = 0; i < grads.size(); ++i)
			{
				grads[i].weights() -= learning_rate_ * grads[i].grad_weights();
				grads[i].bias() -= learning_rate_ * grads[i].grad_bias();
			}

			return logits;
		}
	}
}

