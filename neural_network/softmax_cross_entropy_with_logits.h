#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct softmax_cross_entropy_with_logits
	{
		softmax_cross_entropy_with_logits() = default;
		softmax_cross_entropy_with_logits(tensor_precision loss_precision, tensor_precision output_grads_precision);
		float forward(tensor<float> const& x, tensor<float> const& y) const;
		tensor<float> backward_inputs(tensor<float> const& x, tensor<float> const& y) const;
		backward_grads<float,float,float,float> backward_params() const;
		bool const has_params = false;

	private:
		bool use_precision_{ false };
		tensor_precision loss_precision_;
		tensor_precision output_grads_precision_;
	};

	inline softmax_cross_entropy_with_logits::softmax_cross_entropy_with_logits(tensor_precision loss_precision, tensor_precision output_grads_precision) : 
	use_precision_(true), loss_precision_(loss_precision), output_grads_precision_(output_grads_precision)
	{
	}

	inline float softmax_cross_entropy_with_logits::forward(tensor<float> const& x, tensor<float> const& y) const
	{
		float alpha = 1.f;
		float beta = 0.f;
		auto const n = x.shape().front();

		if(use_precision_)
		{
			auto probs = tensor<float>(y.shape(), loss_precision_);
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto tensor_loss_descriptor = convolution_extensions::make_tensor_descriptor(probs);
			CHECK_CUDNN_ERROR(cudnnSoftmaxForward(
				tensor_lib_init::cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha, *tensor_input_descriptor, x.data(), &beta, *tensor_loss_descriptor, probs.data()));
			return dnn_extensions::cross_entropy_loss<float, float, float>(probs, y, loss_precision_);
		}
		else
		{
			auto probs = tensor<float>(y.shape());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto tensor_loss_descriptor = convolution_extensions::make_tensor_descriptor(probs);
			CHECK_CUDNN_ERROR(cudnnSoftmaxForward(
				tensor_lib_init::cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, 
				&alpha, *tensor_input_descriptor, x.data(), &beta, *tensor_loss_descriptor, probs.data()));
			return dnn_extensions::cross_entropy_loss<float,float,float>(probs, y);
		}
	}

	inline tensor<float> softmax_cross_entropy_with_logits::backward_inputs(tensor<float> const& x, tensor<float> const& y) const
	{
		float alpha = 1.f;
		float beta = 0.f;
		auto const n = x.shape().front();
		
		if (use_precision_)
		{
			tensor<float> output(x.shape(), output_grads_precision_);
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto tensor_label_descriptor = convolution_extensions::make_tensor_descriptor(y);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			CHECK_CUDNN_ERROR(cudnnSoftmaxForward(
				tensor_lib_init::cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha, *tensor_input_descriptor, x.data(), &beta, *tensor_output_descriptor, output.data()));
			tensor_extensions::geam(1.f / n, output, false, -1.f / n, y, false, output);
			return std::move(output);
		}
		else
		{
			tensor<float> output(x.shape());
			auto tensor_input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto tensor_label_descriptor = convolution_extensions::make_tensor_descriptor(y);
			auto tensor_output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			CHECK_CUDNN_ERROR(cudnnSoftmaxForward(
				tensor_lib_init::cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha, *tensor_input_descriptor, x.data(), &beta, *tensor_output_descriptor, output.data()));
			tensor_extensions::geam(1.f / n, output, false, -1.f / n, y, false, output);
			return std::move(output);
		}
	}

	inline backward_grads<float, float, float, float> softmax_cross_entropy_with_logits::backward_params() const
	{
		throw std::runtime_error("Function not implemented!");
	}
}
