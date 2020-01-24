#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct relu_layer : layer
	{
		tensor<float> forward(tensor<float>&& inputs) override;
		tensor<float> forward_inputs(tensor<float>&& inputs) override;
		tensor<float> backward_inputs(tensor<float>&& grads) override;
		backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) override;

		static relu_layer make_relu_layer(
			std::string name);
		static relu_layer make_relu_layer(
			std::string name,
			tensor_precision inputs_grads_precision);

	private:
		relu_layer() = default;

		tensor<float> x_;
		tensor<float> scores_;
		bool use_precision_{ false };
		tensor_precision inputs_grads_precision_;
		activation_descriptor activation_;
	};

	inline tensor<float> relu_layer::forward(tensor<float>&& inputs)
	{
		auto alpha = 1.f;
		auto beta = 0.0f;
		if (use_precision_)
		{
			tensor<float> scores(inputs.shape(), inputs.get_precision());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			CHECK_CUDNN_ERROR(cudnnActivationForward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_input_descriptor, inputs.data(), &beta, *tensor_output_descriptor, scores.data()));
			return std::move(scores);
		}
		else
		{
			tensor<float> scores(inputs.shape());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			CHECK_CUDNN_ERROR(cudnnActivationForward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_input_descriptor, inputs.data(), &beta, *tensor_output_descriptor, scores.data()));
			return scores;
		}
	}

	inline tensor<float> relu_layer::forward_inputs(tensor<float>&& inputs)
	{
		auto alpha = 1.f;
		auto beta = 0.0f;
		if (use_precision_)
		{
			x_ = std::move(inputs);
			scores_ = tensor<float>(x_.shape(), x_.get_precision());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores_);
			CHECK_CUDNN_ERROR(cudnnActivationForward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_input_descriptor, x_.data(), &beta, *tensor_output_descriptor, scores_.data()));
			return scores_;
		}
		else
		{
			x_ = std::move(inputs);
			scores_ = tensor<float>(x_.shape());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores_);
			CHECK_CUDNN_ERROR(cudnnActivationForward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_input_descriptor, x_.data(), &beta, *tensor_output_descriptor, scores_.data()));
			return scores_;
		}
	}

	inline tensor<float> relu_layer::backward_inputs(tensor<float>&& grads)
	{
		auto alpha = 1.f;
		auto beta = 0.0f;
		if (use_precision_)
		{
			auto inputs = std::move(x_);
			auto scores = std::move(scores_);
			auto result = tensor<float>(std::move(grads));
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			auto tensor_output_grads_descriptor = convolution_extensions::make_tensor_descriptor(result);
			CHECK_CUDNN_ERROR(cudnnActivationBackward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_output_descriptor, scores.data(), *tensor_output_grads_descriptor, result.data(),
				*tensor_input_descriptor, inputs.data(), &beta,
				*tensor_output_grads_descriptor, result.data()));
			return tensor<float>(std::move(result), inputs_grads_precision_);
		}
		else
		{
			auto inputs = std::move(x_);
			auto scores = std::move(scores_);
			auto result = tensor<float>(std::move(grads));
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			auto tensor_output_grads_descriptor = convolution_extensions::make_tensor_descriptor(result);
			CHECK_CUDNN_ERROR(cudnnActivationBackward(
				tensor_lib_init::cudnn_handle(), *activation_, &alpha,
				*tensor_output_descriptor, scores.data(), *tensor_output_grads_descriptor, result.data(),
				*tensor_input_descriptor, inputs.data(), &beta,
				*tensor_output_grads_descriptor, result.data()));
			return result;
		}
	}

	backward_grads<float, float, float, float> relu_layer::backward_params(tensor<float> const& grads)
	{
		throw std::runtime_error("Function not implemented!");
	}

	inline relu_layer relu_layer::make_relu_layer(std::string name)
	{
		relu_layer layer;
		layer.has_params = false;
		layer.name = std::move(name);
		layer.activation_ = convolution_extensions::make_activation_descriptor(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

		return layer;
	}

	inline relu_layer relu_layer::make_relu_layer(
		std::string name,
		tensor_precision inputs_grads_precision)
	{
		relu_layer layer;
		layer.has_params = false;
		layer.name = std::move(name);
		layer.inputs_grads_precision_ = inputs_grads_precision;
		layer.use_precision_ = true;
		layer.activation_ = convolution_extensions::make_activation_descriptor(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

		return layer;
	}
}
