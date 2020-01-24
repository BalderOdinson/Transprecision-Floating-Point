#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct flatten_layer : layer
	{
		tensor<float> forward(tensor<float>&& inputs) override;
		tensor<float> forward_inputs(tensor<float>&& inputs) override;
		tensor<float> backward_inputs(tensor<float>&& grads) override;
		backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) override;

		static flatten_layer make_flatten_layer(std::string name);
	private:
		flatten_layer() = default;

		tensor_shape shape_;
	};

	inline tensor<float> flatten_layer::forward(tensor<float>&& inputs)
	{
		auto output = std::move(inputs);
		auto const n = output.shape().front();
		auto const total_size = tensor_shape_total_size(output.shape());
		output.reshape(tensor_shape({ n, total_size / n }));
		return std::move(output);
	}

	inline tensor<float> flatten_layer::forward_inputs(tensor<float>&& inputs)
	{
		auto output = std::move(inputs);
		shape_ = output.shape();
		auto const n = shape_.front();
		auto const total_size = tensor_shape_total_size(shape_);
		output.reshape(tensor_shape({ n, total_size / n }));
		return std::move(output);
	}

	inline tensor<float> flatten_layer::backward_inputs(tensor<float>&& grads)
	{
		auto input_grads = std::move(grads);
		input_grads.reshape(shape_);
		return input_grads;
	}

	inline backward_grads<float, float, float, float> flatten_layer::backward_params(tensor<float> const& grads)
	{
		throw std::runtime_error("Function not implemented!");
	}

	inline flatten_layer flatten_layer::make_flatten_layer(std::string name)
	{
		flatten_layer layer;
		layer.name = std::move(name);
		layer.has_params = false;
		return layer;
	}
}
