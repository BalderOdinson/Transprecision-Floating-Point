#pragma once
#include "layer.h"
#include <functional>
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{

		template<typename Type>
		struct gradient_checker
		{
			template<typename Function>
			tensor<Type> eval_numerical_gradient(Function function, tensor<Type>& x, tensor<Type> const& df, Type h);
			template<typename Function>
			tensor<Type> eval_numerical_gradient(Function function, tensor<Type>& x, Type df, Type h);
			template<typename Layer>
			void check_grad_inputs(Layer& layer, tensor<Type>& x, tensor<Type> const& grad_out);
			template<typename Layer>
			void check_grad_params(Layer& layer, tensor<Type>& x, tensor<Type>& w, tensor<Type>& b, tensor<Type> const& grad_out);

			static Type rel_error(tensor<Type> const& x, tensor<Type> const& y);
		};

		template <typename Type>
		template <typename Function>
		tensor<Type> gradient_checker<Type>::eval_numerical_gradient(Function function,
			tensor<Type>& x, tensor<Type> const& df, Type h)
		{
			tensor<Type> grads(x.shape());
			for (size_t i = 0; i < x.shape().first; ++i)
			{
				for (size_t j = 0; j < x.shape().second; ++j)
				{
					auto const old_value = x[{i, j}];
					x[{i, j}] = old_value + h;
					auto pos = function(x);
					x[{i, j}] = old_value - h;
					auto neg = function(x);

					grads[{i, j}] = ((pos - neg) * df).sum() / (2 * h);
				}
			}

			return grads;
		}

		template <typename Type>
		template <typename Function>
		tensor<Type> gradient_checker<Type>::eval_numerical_gradient(Function function,
			tensor<Type>& x, Type df, Type h)
		{
			tensor<Type> grads(x.shape());
			for (size_t i = 0; i < x.shape().first; ++i)
			{
				for (size_t j = 0; j < x.shape().second; ++j)
				{
					auto const old_value = x[{i, j}];
					x[{i, j}] = old_value + h;
					auto pos = function(x);
					x[{i, j}] = old_value - h;
					auto neg = function(x);

					grads[{i, j}] = ((pos - neg) * df) / (2 * h);
				}
			}

			return grads;
		}

		template <typename Type>
		template <typename Layer>
		void gradient_checker<Type>::check_grad_inputs(Layer& layer, tensor<Type>& x,
			tensor<Type> const& grad_out)
		{
			auto grad_x_num = eval_numerical_gradient(std::bind(&Layer::forward, &layer, std::placeholders::_1), x, grad_out, Type(1e-5));
			auto grad_x = layer.backward_inputs(grad_out);
			std::cout << "Relative error = " << rel_error(grad_x_num, grad_x) << std::endl;
		}

		template <typename Type>
		template <typename Layer>
		void gradient_checker<Type>::check_grad_params(Layer& layer, tensor<Type>& x,
			tensor<Type>& w, tensor<Type>& b, tensor<Type> const& grad_out)
		{
			auto func = [&layer, &x](auto const& params) { return layer.forward(x); };
			auto grad_w_num = eval_numerical_gradient(func, w, grad_out, Type(1e-5));
			auto grad_b_num = eval_numerical_gradient(func, b, grad_out, Type(1e-5));
			auto grads = layer.backward_params(grad_out);
			auto grad_w = grads.grad_weights();
			auto grad_b = grads.grad_bias();
			std::cout << "Check weights :\nRelative error = " << rel_error(grad_w_num, grad_w) << std::endl;
			std::cout << "Check bias :\nRelative error = " << rel_error(grad_b_num, grad_b) << std::endl;
		}

		template <typename Type>
		Type gradient_checker<Type>::rel_error(tensor<Type> const& x, tensor<Type> const& y)
		{
			return (abs(x - y) / maximum(abs(x) + abs(y), Type(1e-8))).max();
		}
	}
}
