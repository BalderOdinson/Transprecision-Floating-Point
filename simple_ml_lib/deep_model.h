#pragma once
#include <array>
#include <iostream>
#include "layer.h"
#include <memory>
#include <unordered_map>
#include <functional>
#include <fstream>
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{

		template<typename Type>
		using model_layer = std::unique_ptr<layer<Type>>;

		template<typename Type, typename Loss>
		struct deep_model
		{
			explicit deep_model(Loss loss);
			deep_model& operator+=(model_layer<Type>&& layer);
			deep_model& operator-=(std::string layer_name);
			tensor<Type> feed_forward(tensor<Type> const& x);
			std::vector<backward_grads<Type>> backward_pass(tensor<Type> const& x, tensor<Type> const& y_oh);
			Type loss(tensor<Type> const& logits, tensor<Type> const& y_oh);
			double accuracy(tensor<Type> const& logits, tensor<Type> const& y_oh);

			tensor<size_t> evaluate(tensor<Type> const& x);
		private:
			Loss loss_;
			std::unordered_map<std::string, size_t> layer_dict_;
			std::unordered_map<size_t, model_layer<Type>> layers_;
			size_t depth_{ 0 };
		};

		template <typename Type, typename Loss>
		deep_model<Type, Loss>::deep_model(Loss loss) : loss_(loss)
		{
		}

		template <typename Type, typename Loss>
		deep_model<Type, Loss>& deep_model<Type, Loss>::operator+=(model_layer<Type>&& layer)
		{
			layer_dict_[layer->name] = depth_;
			layers_[depth_] = std::move(layer);
			++depth_;

			return *this;
		}

		template <typename Type, typename Loss>
		deep_model<Type, Loss>& deep_model<Type, Loss>::operator-=(std::string layer_name)
		{
			if (layer_dict_.find(layer_name) == layer_dict_.end())
				throw std::runtime_error("Layer with given name doesn't exists in current model!");

			auto const index = layer_dict_[layer_name];
			for (auto i = index; i < depth_ - 1; ++i)
			{
				layers_[i] = layers_[i + 1];
				layer_dict_[layers_[i]->name] = i;
			}

			layers_.erase(depth_ - 1);
			layer_dict_.erase(layer_name);
			--depth_;

			return *this;
		}

		template <typename Type, typename Loss>
		tensor<Type> deep_model<Type, Loss>::feed_forward(tensor<Type> const& x)
		{
			auto output = x;
			for (size_t i = 0; i < depth_; ++i)
				output = std::move(layers_[i]->forward(output));

			return output;
		}

		template <typename Type, typename Loss>
		std::vector<backward_grads<Type>> deep_model<Type, Loss>::backward_pass(tensor<Type> const& x,
			tensor<Type> const& y_oh)
		{
			std::vector<backward_grads<Type>> grads;
			auto grads_out = loss_.backward_inputs(x, y_oh);
			if (loss_.has_params)
				grads.push_back(loss_.backward_params());
			for (int64_t i = depth_ - 1; i >= 0; --i)
			{
				auto grad_inputs = layers_[i]->backward_inputs(grads_out);
				if (layers_[i]->has_params)
					grads.push_back(layers_[i]->backward_params(grads_out));
				grads_out = grad_inputs;
			}

			return grads;
		}

		template <typename Type, typename Loss>
		Type deep_model<Type, Loss>::loss(tensor<Type> const& logits, tensor<Type> const& y_oh)
		{
			return loss_.forward(logits, y_oh);
		}

		template <typename Type, typename Loss>
		double deep_model<Type, Loss>::accuracy(tensor<Type> const& logits,
			tensor<Type> const& y_oh)
		{
			auto y = logits.template argmax<1>().template cast<double>();
			auto y_ = y_oh.template argmax<1>().template cast<double>();

			double const n = y.shape().first;

			return produce(y, y_, [] DEVICE_FUNCTION(Type y_f, Type y_f_) { return y_f == y_f_ ? Type(1) : Type(0); }).sum() / n;
		}

		template <typename Type, typename Loss>
		tensor<size_t> deep_model<Type, Loss>::evaluate(tensor<Type> const& x)
		{
			return feed_forward(x).template argmax<1>();
		}

#ifdef USE_CUDA
#include "../cuda_blas/distributions.h"
		template<typename Type>
		tensor<Type> uniform_real_random_initializer(tensor_shape shape)
		{
			return tensor<Type>::template create_random<transprecision_floating_point::cuda_blas::uniform_distribution>(shape.first, shape.second, cuda_blas::unifrom_distribution(0, 1));
		}

		template<typename Type>
		tensor<Type> zeros_initializer(tensor_shape shape)
		{
			return tensor<Type>(shape);
		}

		template<typename Type>
		struct variance_normal_initializer
		{
			variance_normal_initializer(Type fan_in) : fan_in_(fan_in) {  }
			tensor<Type> operator()(tensor_shape shape)
			{
				
				return tensor<Type>::template create_random<transprecision_floating_point::cuda_blas::variance_normal_initializer>(shape.first, shape.second, cuda_blas::variance_normal_initializer(fan_in_));
			}

		private:
			Type fan_in_;
		};
#else
		template<typename Type>
		tensor<Type> uniform_real_random_initializer(tensor_shape shape)
		{
			return tensor<Type>::template create_random<std::uniform_real_distribution<float>>(shape.first, shape.second, std::uniform_real_distribution<float>(0, 1));
		}

		template<typename Type>
		tensor<Type> zeros_initializer(tensor_shape shape)
		{
			return tensor<Type>(shape);
		}

		template<typename Type>
		struct variance_normal_initializer
		{
			variance_normal_initializer(Type fan_in) : fan_in_(fan_in) {  }
			tensor<Type> operator()(tensor_shape shape)
			{
				Type sigma = sqrtf(2 / static_cast<float>(fan_in_));
				return tensor<Type>::template create_random<std::normal_distribution<float>>(shape.first, shape.second, std::normal_distribution<float>()) * sigma;
			}

		private:
			Type fan_in_;
		};
#endif
	}
}

