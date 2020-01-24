#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct max_pool_layer : layer
	{
		tensor<float> forward(tensor<float>&& inputs) override;
		tensor<float> forward_inputs(tensor<float>&& inputs) override;
		tensor<float> backward_inputs(tensor<float>&& grads) override;
		backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) override;

		static max_pool_layer make_max_pool_layer(
			size_t height, size_t width,
			size_t vertical_stride, size_t horizontal_stride,
			std::string name);
		static max_pool_layer make_max_pool_layer(
			size_t height, size_t width,
			size_t vertical_stride, size_t horizontal_stride,
			std::string name,
			tensor_precision inputs_grads_precision);

	private:
		max_pool_layer() = default;

		pooling_descriptor pooling_descriptor_;
		size_t vertical_stride_; 
		size_t horizontal_stride_;
		tensor<float> x_;
		tensor<float> scores_;
		bool use_precision_{ false };
		tensor_precision inputs_grads_precision_;
	};

	inline tensor<float> max_pool_layer::forward(tensor<float>&& inputs)
	{
		auto output_shape = inputs.shape();
		output_shape.back() /= horizontal_stride_;
		output_shape[output_shape.size() - 2] /= vertical_stride_;
		tensor<float> scores(output_shape, inputs.get_precision());
		auto inputs_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
		auto output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
		convolution_extensions::pooling_forward(pooling_descriptor_, inputs_descriptor, inputs, output_descriptor, scores);

		return std::move(scores);
	}

	inline tensor<float> max_pool_layer::forward_inputs(tensor<float>&& inputs)
	{
		x_ = std::move(inputs);
		auto output_shape = x_.shape();
		output_shape.back() /= horizontal_stride_;
		output_shape[output_shape.size() - 2] /= vertical_stride_;
		scores_ = tensor<float>(output_shape, x_.get_precision());
		auto inputs_descriptor = convolution_extensions::make_tensor_descriptor(x_);
		auto output_descriptor = convolution_extensions::make_tensor_descriptor(scores_);
		convolution_extensions::pooling_forward(pooling_descriptor_, inputs_descriptor, x_, output_descriptor, scores_);

		return scores_;
	}

	inline tensor<float> max_pool_layer::backward_inputs(tensor<float>&& grads)
	{
		if(use_precision_)
		{
			auto inputs = std::move(x_);
			auto scores = std::move(scores_);
			tensor<float> input_grads(inputs.shape());
			auto inputs_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			auto grads_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			auto input_grads_descriptor = convolution_extensions::make_tensor_descriptor(input_grads);

			convolution_extensions::pooling_backward(
				pooling_descriptor_,
				output_descriptor, scores,
				grads_descriptor, grads,
				inputs_descriptor, inputs,
				input_grads_descriptor, input_grads);

			return tensor<float>(std::move(input_grads), inputs_grads_precision_);
		}
		else
		{
			auto inputs = std::move(x_);
			auto scores = std::move(scores_);
			tensor<float> input_grads(inputs.shape());
			auto inputs_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(scores);
			auto grads_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			auto input_grads_descriptor = convolution_extensions::make_tensor_descriptor(input_grads);

			convolution_extensions::pooling_backward(
				pooling_descriptor_,
				output_descriptor, scores,
				grads_descriptor, grads,
				inputs_descriptor, inputs,
				input_grads_descriptor, input_grads);

			return input_grads;
		}
	}

	inline backward_grads<float, float, float, float> max_pool_layer::backward_params(tensor<float> const& grads)
	{
		throw std::runtime_error("Function not implemented!");
	}

	inline max_pool_layer max_pool_layer::make_max_pool_layer(size_t height, size_t width, size_t vertical_stride,
		size_t horizontal_stride, std::string name)
	{
		max_pool_layer layer;
		layer.has_params = false;
		layer.name = std::move(name);
		layer.vertical_stride_ = vertical_stride;
		layer.horizontal_stride_ = horizontal_stride;
		layer.pooling_descriptor_ = convolution_extensions::make_2d_pooling_descriptor(CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, height, width, 0, 0, vertical_stride, horizontal_stride);

		return layer;
	}

	inline max_pool_layer max_pool_layer::make_max_pool_layer(size_t height, size_t width, size_t vertical_stride,
		size_t horizontal_stride, std::string name, tensor_precision inputs_grads_precision)
	{
		max_pool_layer layer;
		layer.has_params = false;
		layer.name = std::move(name);
		layer.vertical_stride_ = vertical_stride;
		layer.horizontal_stride_ = horizontal_stride;
		layer.pooling_descriptor_ = convolution_extensions::make_2d_pooling_descriptor(CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, height, width, 0, 0, vertical_stride, horizontal_stride);
		layer.inputs_grads_precision_ = inputs_grads_precision;
		layer.use_precision_ = true;

		return layer;
	}
}
