#pragma once
#include "layer.h"
#include "shared_workspace_memory.h"

namespace transprecision_floating_point
{
	template<typename T>
	struct conv2d_layer : layer
	{
		template<typename WeightsInitializer, typename BiasInitializer>
		static conv2d_layer make_conv2d_layer(
			size_t in_channels, size_t out_channels,
			size_t kernel_height, size_t kernel_width,
			size_t vertical_stride, size_t horizontal_stride,
			WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name);
		template<typename WeightsInitializer, typename BiasInitializer>
		static conv2d_layer make_conv2d_layer(
			size_t in_channels, size_t out_channels,
			size_t kernel_height, size_t kernel_width,
			size_t vertical_stride, size_t horizontal_stride,
			WeightsInitializer weights_initializer, BiasInitializer bias_initializer, std::string name,
			tensor_precision output_precision, tensor_precision weights_precision,
			tensor_precision bias_precision, tensor_precision weight_grads_precision, tensor_precision bias_grads_precision,
			tensor_precision inputs_grads_precision);

		tensor<float> forward(tensor<float>&& inputs) override;
		tensor<float> forward_inputs(tensor<float>&& inputs) override;
		tensor<float> backward_inputs(tensor<float>&& grads) override;
		backward_grads<float, float, float, float> backward_params(tensor<float> const& grads) override;

		tensor<float> const& get_weights() const;
		tensor<float> const& get_bias() const;

	private:
		conv2d_layer() = default;

		size_t in_channels_;
		size_t out_channels_;
		size_t kernel_height_;
		size_t kernel_width_;
		size_t vertical_stride_;
		size_t horizontal_stride_;
		tensor<T> x_;
		tensor<float> weights_;
		tensor<float> bias_;
		filter_descriptor weight_descriptor_;
		convolution_descriptor convolution_descriptor_;
		bool use_precision_{ false };
		tensor_precision output_precision_;
		tensor_precision weights_precision_;
		tensor_precision bias_precision_;
		tensor_precision weight_grads_precision_;
		tensor_precision bias_grads_precision_;
		tensor_precision inputs_grads_precision_;
	};

	template <typename T>
	template <typename WeightsInitializer, typename BiasInitializer>
	conv2d_layer<T> conv2d_layer<T>::make_conv2d_layer(size_t in_channels, size_t out_channels, size_t kernel_height,
		size_t kernel_width, size_t vertical_stride, size_t horizontal_stride, WeightsInitializer weights_initializer,
		BiasInitializer bias_initializer, std::string name)
	{
		conv2d_layer<T> layer;
		layer.has_params = true;
		layer.name = std::move(name);
		layer.in_channels_ = in_channels;
		layer.out_channels_ = out_channels;
		layer.kernel_height_ = kernel_height;
		layer.kernel_width_ = kernel_width;
		layer.vertical_stride_ = vertical_stride;
		layer.horizontal_stride_ = horizontal_stride;
		layer.weights_ = tensor<float>(std::move(weights_initializer(tensor_shape({ out_channels, in_channels, kernel_height, kernel_width }))));
		layer.bias_ = tensor<float>(std::move(bias_initializer(tensor_shape({ 1, out_channels, 1, 1, }))));
		layer.weight_descriptor_ = convolution_extensions::make_4d_filter_descriptor<T>(out_channels, in_channels, kernel_height, kernel_width);
		layer.convolution_descriptor_ = convolution_extensions::make_2d_convolution_descriptor<T>(0, 0, vertical_stride, horizontal_stride, 1, 1, CUDNN_CROSS_CORRELATION);

		return layer;
	}

	template <typename T>
	template <typename WeightsInitializer, typename BiasInitializer>
	conv2d_layer<T> conv2d_layer<T>::make_conv2d_layer(size_t in_channels, size_t out_channels, size_t kernel_height,
		size_t kernel_width, size_t vertical_stride, size_t horizontal_stride, WeightsInitializer weights_initializer,
		BiasInitializer bias_initializer, std::string name, tensor_precision output_precision,
		tensor_precision weights_precision, tensor_precision bias_precision, tensor_precision weight_grads_precision,
		tensor_precision bias_grads_precision, tensor_precision inputs_grads_precision)
	{
		conv2d_layer<T> layer;
		layer.has_params = true;
		layer.name = std::move(name);
		layer.in_channels_ = in_channels;
		layer.out_channels_ = out_channels;
		layer.kernel_height_ = kernel_height;
		layer.kernel_width_ = kernel_width;
		layer.vertical_stride_ = vertical_stride;
		layer.horizontal_stride_ = horizontal_stride;
		layer.weights_ = tensor<float>(std::move(weights_initializer(tensor_shape({ out_channels, in_channels, kernel_height, kernel_width }))), weights_precision);
		layer.bias_ = tensor<float>(std::move(bias_initializer(tensor_shape({ 1, out_channels, 1, 1, }))), bias_precision);
		layer.weight_descriptor_ = convolution_extensions::make_4d_filter_descriptor<T>(out_channels, in_channels, kernel_height, kernel_width);
		layer.convolution_descriptor_ = convolution_extensions::make_2d_convolution_descriptor<T>(0, 0, vertical_stride, horizontal_stride, 1, 1, CUDNN_CROSS_CORRELATION);
		layer.output_precision_ = output_precision;
		layer.weights_precision_ = weights_precision;
		layer.bias_precision_ = bias_precision;
		layer.weight_grads_precision_ = weight_grads_precision;
		layer.bias_grads_precision_ = bias_grads_precision;
		layer.inputs_grads_precision_ = inputs_grads_precision;
		layer.use_precision_ = true;

		return layer;
	}

	template <typename T>
	tensor<float> conv2d_layer<T>::forward(tensor<float>&& inputs)
	{
		if (use_precision_)
		{
			tensor<T> x(inputs);
			tensor<T> weights(weights_);
			tensor<T> bias(bias_);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias);
			tensor<T> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_alg = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_alg);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x,
				weight_descriptor_, weights,
				convolution_descriptor_, forward_alg,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias);

			return tensor<float>(output, output_precision_);
		}
		else
		{
			tensor<T> x(inputs);
			tensor<T> weights(weights_);
			tensor<T> bias(bias_);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias);
			tensor<T> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_algo = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x,
				weight_descriptor_, weights,
				convolution_descriptor_, forward_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias);

			return tensor<float>(output);
		}
	}

	template <>
	inline tensor<float> conv2d_layer<float>::forward(tensor<float>&& inputs)
	{
		if (use_precision_)
		{
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias_);
			tensor<float> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_alg = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_alg);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, inputs,
				weight_descriptor_, weights_,
				convolution_descriptor_, forward_alg,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias_);

			return tensor<float>(std::move(output), output_precision_);
		}
		else
		{
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias_);
			tensor<float> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_algo = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, inputs,
				weight_descriptor_, weights_,
				convolution_descriptor_, forward_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias_);

			return std::move(output);
		}
	}

	template <typename T>
	tensor<float> conv2d_layer<T>::forward_inputs(tensor<float>&& inputs)
	{
		if(use_precision_)
		{
			x_ = std::move(inputs);
			tensor<T> weights(weights_);
			tensor<T> bias(bias_);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias);
			tensor<T> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_alg = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_alg);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x_,
				weight_descriptor_, weights,
				convolution_descriptor_, forward_alg,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias);

			return tensor<float>(output, output_precision_);
		}
		else
		{
			x_ = std::move(inputs);
			tensor<T> weights(weights_);
			tensor<T> bias(bias_);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias);
			tensor<T> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_algo = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_algo);
			if(new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x_,
				weight_descriptor_, weights,
				convolution_descriptor_, forward_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias);

			return tensor<float>(output);
		}
	}

	template <>
	inline tensor<float> conv2d_layer<float>::forward_inputs(tensor<float>&& inputs)
	{
		if (use_precision_)
		{
			x_ = std::move(inputs);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias_);
			tensor<float> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_alg = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_alg);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x_,
				weight_descriptor_, weights_,
				convolution_descriptor_, forward_alg,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias_);

			return tensor<float>(std::move(output), output_precision_);
		}
		else
		{
			x_ = std::move(inputs);
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto bias_descriptor = convolution_extensions::make_tensor_descriptor(bias_);
			tensor<float> output(convolution_extensions::get_conv2_forward_output_shape(convolution_descriptor_, input_descriptor, weight_descriptor_));
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(output);
			auto forward_algo = convolution_extensions::get_convolution_forward_algorithm(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_forward_workspace_size(convolution_descriptor_, input_descriptor, weight_descriptor_, output_descriptor, forward_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_forward(
				input_descriptor, x_,
				weight_descriptor_, weights_,
				convolution_descriptor_, forward_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				output_descriptor, output,
				bias_descriptor, bias_);

			return std::move(output);
		}
	}

	template <typename T>
	tensor<float> conv2d_layer<T>::backward_inputs(tensor<float>&& grads)
	{
		if (use_precision_)
		{
			auto inputs = std::move(x_);
			tensor<T> weights(weights_);
			tensor<T> grads_t(std::move(grads));
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads_t);
			auto backward_data_algo = convolution_extensions::get_convolution_backward_data_algorithm(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_data_workspace_size(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor, backward_data_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_data(
				weight_descriptor_, weights,
				output_descriptor, grads_t,
				convolution_descriptor_, backward_data_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				input_descriptor, inputs);

			return tensor<float>(inputs, inputs_grads_precision_);
		}
		else
		{
			auto inputs = std::move(x_);
			tensor<T> weights(weights_);
			tensor<T> grads_t(std::move(grads));
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(inputs);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads_t);
			auto backward_data_algo = convolution_extensions::get_convolution_backward_data_algorithm(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_data_workspace_size(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor, backward_data_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_data(
				weight_descriptor_, weights,
				output_descriptor, grads_t,
				convolution_descriptor_, backward_data_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				input_descriptor, inputs);

			return tensor<float>(inputs);
		}
	}

	template <>
	inline tensor<float> conv2d_layer<float>::backward_inputs(tensor<float>&& grads)
	{
		if (use_precision_)
		{
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			auto backward_data_algo = convolution_extensions::get_convolution_backward_data_algorithm(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_data_workspace_size(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor, backward_data_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_data(
				weight_descriptor_, weights_,
				output_descriptor, grads,
				convolution_descriptor_, backward_data_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				input_descriptor, x_);

			return tensor<float>(std::move(x_), inputs_grads_precision_);
		}
		else
		{

			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			auto backward_data_algo = convolution_extensions::get_convolution_backward_data_algorithm(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_data_workspace_size(weight_descriptor_, output_descriptor, convolution_descriptor_, input_descriptor, backward_data_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_data(
				weight_descriptor_, weights_,
				output_descriptor, grads,
				convolution_descriptor_, backward_data_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				input_descriptor, x_);

			return std::move(x_);
		}
	}

	template <typename T>
	backward_grads<float, float, float, float> conv2d_layer<T>::backward_params(tensor<float> const& grads)
	{
		if (use_precision_)
		{
			tensor<T> grads_t(grads);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads_t);
			tensor<T> grads_bias(bias_.shape());
			auto grads_bias_descriptor = convolution_extensions::make_tensor_descriptor(grads_bias);
			convolution_extensions::convolution_backward_bias(output_descriptor, grads_t, grads_bias_descriptor, grads_bias);

			tensor<T> grads_weight(weights_.shape());
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto backward_filter_algo = convolution_extensions::get_convolution_backward_filter_algorithm(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_filter_workspace_size(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_, backward_filter_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_filter(
				input_descriptor, x_,
				output_descriptor, grads_t,
				convolution_descriptor_, backward_filter_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				weight_descriptor_, grads_weight);

			return { &weights_, tensor<float>(grads_weight, weight_grads_precision_), &bias_, tensor<float>(grads_bias, bias_grads_precision_) };
		}
		else
		{
			tensor<T> grads_t(grads);
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads_t);
			tensor<T> grads_bias(bias_.shape(), bias_.get_precision());
			auto grads_bias_descriptor = convolution_extensions::make_tensor_descriptor(grads_bias);
			convolution_extensions::convolution_backward_bias(output_descriptor, grads_t, grads_bias_descriptor, grads_bias);

			tensor<T> grads_weight(weights_.shape(), weights_.get_precision());
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto backward_filter_algo = convolution_extensions::get_convolution_backward_filter_algorithm(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_filter_workspace_size(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_, backward_filter_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_filter(
				input_descriptor, x_,
				output_descriptor, grads_t,
				convolution_descriptor_, backward_filter_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				weight_descriptor_, grads_weight);

			return { &weights_, tensor<float>(grads_weight), &bias_, tensor<float>(grads_bias) };
		}
	}

	template <>
	inline backward_grads<float, float, float, float> conv2d_layer<float>::backward_params(tensor<float> const& grads)
	{
		if (use_precision_)
		{
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			tensor<float> grads_bias(bias_.shape());
			auto grads_bias_descriptor = convolution_extensions::make_tensor_descriptor(grads_bias);
			convolution_extensions::convolution_backward_bias(output_descriptor, grads, grads_bias_descriptor, grads_bias);

			tensor<float> grads_weight(weights_.shape());
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto backward_filter_algo = convolution_extensions::get_convolution_backward_filter_algorithm(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_filter_workspace_size(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_, backward_filter_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_filter(
				input_descriptor, x_,
				output_descriptor, grads,
				convolution_descriptor_, backward_filter_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				weight_descriptor_, grads_weight);

			return { &weights_, tensor<float>(std::move(grads_weight), weight_grads_precision_), &bias_, tensor<float>(std::move(grads_bias), bias_grads_precision_) };
		}
		else
		{
			auto output_descriptor = convolution_extensions::make_tensor_descriptor(grads);
			tensor<float> grads_bias(bias_.shape(), bias_.get_precision());
			auto grads_bias_descriptor = convolution_extensions::make_tensor_descriptor(grads_bias);
			convolution_extensions::convolution_backward_bias(output_descriptor, grads, grads_bias_descriptor, grads_bias);

			tensor<float> grads_weight(weights_.shape(), weights_.get_precision());
			auto input_descriptor = convolution_extensions::make_tensor_descriptor(x_);
			auto backward_filter_algo = convolution_extensions::get_convolution_backward_filter_algorithm(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_);
			auto new_workspace_size = convolution_extensions::get_convolution_backward_filter_workspace_size(input_descriptor, output_descriptor, convolution_descriptor_, weight_descriptor_, backward_filter_algo);
			if (new_workspace_size > shared_workspace_memory::workspace_size)
			{
				shared_workspace_memory::workspace_size = new_workspace_size;
				shared_workspace_memory::memory = convolution_extensions::make_workspace_memory(shared_workspace_memory::workspace_size);
			}

			convolution_extensions::convolution_backward_filter(
				input_descriptor, x_,
				output_descriptor, grads,
				convolution_descriptor_, backward_filter_algo,
				shared_workspace_memory::memory, shared_workspace_memory::workspace_size,
				weight_descriptor_, grads_weight);

			return { &weights_, std::move(grads_weight), &bias_, std::move(grads_bias) };
		}
	}

	template <typename T>
	tensor<float> const& conv2d_layer<T>::get_weights() const
	{
		return weights_;
	}

	template <typename T>
	tensor<float> const& conv2d_layer<T>::get_bias() const
	{
		return bias_;
	}
}
