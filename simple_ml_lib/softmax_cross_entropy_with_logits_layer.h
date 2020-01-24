#pragma once
#include "layer.h"
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct softmax_cross_entropy_with_logits
		{
			explicit softmax_cross_entropy_with_logits(Type epsilon);
			Type forward(tensor<Type> const& x, tensor<Type> const& y);
			tensor<Type> backward_inputs(tensor<Type> const& x, tensor<Type> const& y);
			backward_grads<Type> backward_params() const;
			bool const has_params = false;

		private:
			Type epsilon_;
		};

		template <typename Type>
		softmax_cross_entropy_with_logits<Type>::softmax_cross_entropy_with_logits(Type epsilon) : epsilon_(epsilon)
		{
		}

		template <typename Type>
		Type softmax_cross_entropy_with_logits<Type>::forward(
			tensor<Type> const& x,
			tensor<Type> const& y)
		{
			auto const n = x.shape().first;

			auto const exp_scores = exp(x - x.max()).apply([epsilon = epsilon_] DEVICE_FUNCTION(Type value) { return value == Type(0) ? epsilon : value; });
			auto const sum_exp_scores = exp_scores.template sum<1>();
			auto const probs = (exp_scores / sum_exp_scores);

			auto const log_probs = log(probs);

			return (-1 * (log_probs * y).sum()) / Type(n);
		}

		template <typename Type>
		tensor<Type> softmax_cross_entropy_with_logits<Type>::backward_inputs(
			tensor<Type> const& x,
			tensor<Type> const& y)
		{
			auto const n = x.shape().first;

			auto const exp_scores = exp(x - x.max()).apply([epsilon = epsilon_] DEVICE_FUNCTION(Type value) { return value == Type(0) ? epsilon : value; });
			auto const sum_exp_scores = exp_scores.template sum<1>();
			auto const probs = (exp_scores / sum_exp_scores);

			return (probs - y) / Type(n);
		}

		template <typename Type>
		backward_grads<Type> softmax_cross_entropy_with_logits<Type>::backward_params() const
		{
			throw std::runtime_error("Function not implemented!");
		}
	}
}
