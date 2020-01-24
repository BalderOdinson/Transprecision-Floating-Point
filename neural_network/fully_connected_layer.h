#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct fully_connected_layer : layer
	{
		tensor<float> forward(tensor<float>&& inputs) override;
		tensor<float> forward_inputs(tensor<float>&& inputs) override;
		tensor<float> backward_inputs(tensor<float>&& grads) override;
		backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) override;

		template<typename WeightsInitializer, typename BiasInitializer>
		static fully_connected_layer make_fully_connected_layer(
			size_t input_layer, size_t output_count,
			WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name);
		template<typename WeightsInitializer, typename BiasInitializer>
		static fully_connected_layer make_fully_connected_layer(
			size_t input_layer, size_t output_count,
			WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name,
			tensor_precision output_precision, tensor_precision weights_precision,
			tensor_precision bias_precision, tensor_precision weight_grads_precision, tensor_precision bias_grads_precision,
			tensor_precision inputs_grads_precision);

	private:
		fully_connected_layer() = default;
	
		tensor<float> weights_;
		tensor<float> bias_;
		tensor<float> x_;
		bool use_precision_{ false };
		tensor_precision output_precision_;
		tensor_precision weights_precision_;
		tensor_precision bias_precision_;
		tensor_precision weight_grads_precision_;
		tensor_precision bias_grads_precision_;
		tensor_precision inputs_grads_precision_;
	};

	inline tensor<float> fully_connected_layer::forward(tensor<float>&& inputs)
	{
		if (use_precision_)
		{
			tensor<float> result(bias_, output_precision_);
			tensor_extensions::gemm_ex(1.f, inputs, false, weights_, true, 1.f, false, result);
			return std::move(result);
		}
		else
		{
			tensor<float> result = bias_;
			tensor_extensions::gemm_ex(1.f, inputs, false, weights_, true, 1.f, false, result);
			return std::move(result);
		}
	}

	inline tensor<float> fully_connected_layer::forward_inputs(tensor<float>&& inputs)
	{
		if (use_precision_)
		{
			x_ = std::move(inputs);
			tensor<float> result(bias_, output_precision_);
			tensor_extensions::gemm_ex(1.f, x_, false, weights_, true, 1.f, false, result);
			return std::move(result);
		}
		else
		{
			x_ = std::move(inputs);
			tensor<float> result = bias_;
			tensor_extensions::gemm_ex(1.f, x_, false, weights_, true, 1.f, false, result);
			return std::move(result);
		}
	}

	inline tensor<float> fully_connected_layer::backward_inputs(tensor<float>&& grads)
	{
		if (use_precision_)
			return tensor<float>(grads.dot(weights_), inputs_grads_precision_);
		else
			return grads.dot(weights_);
	}

	inline backward_grads<float, float, float, float> fully_connected_layer::backward_params(tensor<float> const& grads)
	{
		if (use_precision_)
		{
			auto inputs = std::move(x_);
			tensor<float> grad_weights(weights_.shape(), weight_grads_precision_);
			tensor<float> grad_bias(grads.sum(0), bias_grads_precision_);

			tensor_extensions::gemm(1.f, grads, true, inputs, false, 0.f, grad_weights);

			return { &weights_, std::move(grad_weights), &bias_, std::move(grad_bias) };
		}
		else
		{
			auto inputs = std::move(x_);
			tensor<float> grad_weights(weights_.shape());
			tensor<float> grad_bias(grads.sum(0));

			tensor_extensions::gemm(1.f, grads, true, inputs, false, 0.f, grad_weights);

			return { &weights_, std::move(grad_weights), &bias_, std::move(grad_bias) };
		}
	}


	template <typename WeightsInitializer, typename BiasInitializer>
	fully_connected_layer fully_connected_layer::make_fully_connected_layer(size_t input_layer, size_t output_count, WeightsInitializer weights_initializer,
		BiasInitializer bias_initializer, std::string name)
	{
		fully_connected_layer layer;
		layer.has_params = true;
		layer.name = std::move(name);
		layer.weights_ = tensor<float>(std::move(weights_initializer(tensor_shape({ output_count, input_layer }))));
		layer.bias_ = tensor<float>(std::move(bias_initializer(tensor_shape({ 1, output_count }))));

		return layer;
	}


	template <typename WeightsInitializer, typename BiasInitializer>
	fully_connected_layer fully_connected_layer::make_fully_connected_layer(size_t input_layer, size_t output_count, WeightsInitializer weights_initializer,
	                                                                        BiasInitializer bias_initializer, std::string name,
	                                                                        tensor_precision output_precision, tensor_precision weights_precision, tensor_precision bias_precision,
	                                                                        tensor_precision weight_grads_precision, tensor_precision bias_grads_precision,
	                                                                        tensor_precision inputs_grads_precision)
	{
		fully_connected_layer layer;
		layer.has_params = true;
		layer.name = std::move(name);
		layer.weights_ = tensor<float>(std::move(weights_initializer(tensor_shape({ output_count, input_layer }))), weights_precision);
		layer.bias_ = tensor<float>(std::move(bias_initializer(tensor_shape({ 1, output_count }))), bias_precision);
		layer.output_precision_ = output_precision;
		layer.weights_precision_ = weights_precision;
		layer.bias_precision_ = bias_precision;
		layer.weight_grads_precision_ = weight_grads_precision;
		layer.bias_grads_precision_ = bias_grads_precision;
		layer.inputs_grads_precision_ = inputs_grads_precision;
		layer.use_precision_ = true;

		return layer;
	}
}
