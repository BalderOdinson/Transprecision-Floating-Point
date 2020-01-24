#pragma once
#include "tensor.h"
#include <cudnn.h>

namespace transprecision_floating_point
{
	using tensor_descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>;
	using activation_descriptor = std::unique_ptr<cudnnActivationDescriptor_t, std::function<void(cudnnActivationDescriptor_t*)>>;
	using pooling_descriptor = std::unique_ptr<cudnnPoolingDescriptor_t, std::function<void(cudnnPoolingDescriptor_t*)>>;
	using filter_descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>;
	using convolution_descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>;
	using workspace_memory = std::unique_ptr<float[], std::function<void(float*)>>;

	struct convolution_extensions
	{
		template<typename T>
		static tensor_descriptor make_tensor_descriptor(tensor<T> const& tensor);
		static activation_descriptor make_activation_descriptor(cudnnActivationMode_t activation_mode, cudnnNanPropagation_t nan_propagation, double coefficient);
		static pooling_descriptor make_2d_pooling_descriptor(
			cudnnPoolingMode_t pooling_mode, cudnnNanPropagation_t nan_propagation,
			int window_height, int window_width,
			int vertical_padding, int horizontal_padding,
			int vertical_stride, int horizontal_stride);
		template<typename T>
		static filter_descriptor make_4d_filter_descriptor(int k, int c, int h, int w);
		template<typename T>
		static convolution_descriptor make_2d_convolution_descriptor(int padding_height, int padding_width, int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode);
		static workspace_memory make_workspace_memory(size_t size);
		static tensor_shape get_conv2_forward_output_shape(convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor, filter_descriptor const& kernel_descriptor);
		static cudnnConvolutionFwdAlgo_t get_convolution_forward_algorithm(convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor, filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_tensor_descriptor);
		static size_t get_convolution_forward_workspace_size(convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor, filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_tensor_descriptor, cudnnConvolutionFwdAlgo_t algo);
		template<typename I, typename W, typename B, typename O>
		static void convolution_forward(tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input, filter_descriptor const& kernel_descriptor, tensor<W> const& weights, convolution_descriptor const& convolution, cudnnConvolutionFwdAlgo_t algo, workspace_memory const& memory, size_t workspace_size, tensor_descriptor const& output_tensor_descriptor, tensor<O>& output, tensor_descriptor const& bias_tensor_descriptor, tensor<B> const& bias);
		static cudnnConvolutionBwdDataAlgo_t get_convolution_backward_data_algorithm(filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_descriptor, convolution_descriptor const& convolution, tensor_descriptor const& input_descriptor);
		static size_t get_convolution_backward_data_workspace_size(filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_descriptor, convolution_descriptor const& convolution, tensor_descriptor const& input_descriptor, cudnnConvolutionBwdDataAlgo_t algo);
		template<typename I, typename W, typename O>
		static void convolution_backward_data(filter_descriptor const& kernel_descriptor, tensor<W> const& weights, tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output, convolution_descriptor const& convolution, cudnnConvolutionBwdDataAlgo_t algo, workspace_memory const& memory, size_t workspace_size, tensor_descriptor const& input_tensor_descriptor, tensor<I>& input);
		template<typename O, typename B>
		static void convolution_backward_bias(tensor_descriptor const& output_descriptor, tensor<O> const& output, tensor_descriptor const& bias_tensor_descriptor, tensor<B>& grads_bias);
		static cudnnConvolutionBwdFilterAlgo_t get_convolution_backward_filter_algorithm(tensor_descriptor const& input_tensor_descriptor, tensor_descriptor const& output_tensor_descriptor, convolution_descriptor const& convolution, filter_descriptor const& kernel_descriptor);
		static size_t get_convolution_backward_filter_workspace_size(tensor_descriptor const& input_tensor_descriptor, tensor_descriptor const& output_tensor_descriptor, convolution_descriptor const& convolution, filter_descriptor const& kernel_descriptor, cudnnConvolutionBwdFilterAlgo_t algo);
		template<typename I, typename O, typename W>
		static void convolution_backward_filter(tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input, tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output, convolution_descriptor const& convolution, cudnnConvolutionBwdFilterAlgo_t algo, workspace_memory const& memory, size_t workspace_size, filter_descriptor const& kernel_descriptor, tensor<W>& grads_weight);
		template<typename I, typename O>
		static void pooling_forward(pooling_descriptor const& pooling, tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input, tensor_descriptor const& output_tensor_descriptor, tensor<O>& output);
		template<typename I, typename O, typename G, typename R>
		static void pooling_backward(pooling_descriptor const& pooling, tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output, tensor_descriptor const& grads_tensor_descriptor, tensor<G> const& grads, tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input, tensor_descriptor const& result_descriptor, tensor<R>& result);
	};

	template <typename T>
	tensor_descriptor convolution_extensions::make_tensor_descriptor(tensor<T> const& tensor)
	{
		tensor_descriptor descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>(new cudnnTensorDescriptor_t(),
			[](cudnnTensorDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(descriptor.get()));

		auto dim = tensor_shape_dimension(tensor.shape_);
		if (dim < 5)
		{
			size_t n = 1, c = 1, h = 1, w = 1;

			for (size_t i = 0; i < dim; ++i)
			{
				switch (i % 4)
				{
				case 0:
					n = tensor.shape_[i];
					break;
				case 1:
					c = tensor.shape_[i];
					break;
				case 2:
					h = tensor.shape_[i];
					break;
				case 3:
					w = tensor.shape_[i];
					break;
				}
			}

			CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
		}
		else if (dim <= CUDNN_DIM_MAX)
		{
			std::vector<int> dims;
			for (size_t i = 0; i < dim; ++i)
				dims[i] = tensor.shape_[i];
			CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptorEx(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dim, dims.data()));
		}
		else
		{
			throw std::runtime_error("Invalid tensor dimension!");
		}

		return descriptor;
	}

	template <>
	inline tensor_descriptor convolution_extensions::make_tensor_descriptor(tensor<half> const& tensor)
	{
		tensor_descriptor descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>(new cudnnTensorDescriptor_t(),
			[](cudnnTensorDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(descriptor.get()));

		auto dim = tensor_shape_dimension(tensor.shape_);
		if (dim < 5)
		{
			size_t n = 1, c = 1, h = 1, w = 1;

			for (size_t i = 0; i < dim; ++i)
			{
				switch (i % 4)
				{
				case 0:
					n = tensor.shape_[i];
					break;
				case 1:
					c = tensor.shape_[i];
					break;
				case 2:
					h = tensor.shape_[i];
					break;
				case 3:
					w = tensor.shape_[i];
					break;
				}
			}

			CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
		}
		else if (dim <= CUDNN_DIM_MAX)
		{
			std::vector<int> dims;
			for (size_t i = 0; i < dim; ++i)
				dims[i] = tensor.shape_[i];
			CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptorEx(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, dim, dims.data()));
		}
		else
		{
			throw std::runtime_error("Invalid tensor dimension!");
		}

		return descriptor;
	}

	template <>
	inline tensor_descriptor convolution_extensions::make_tensor_descriptor(tensor<int32_t> const& tensor)
	{
		tensor_descriptor descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>(new cudnnTensorDescriptor_t(),
			[](cudnnTensorDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(descriptor.get()));

		auto dim = tensor_shape_dimension(tensor.shape_);
		if (dim < 5)
		{
			size_t n = 1, c = 1, h = 1, w = 1;

			for (size_t i = 0; i < dim; ++i)
			{
				switch (i % 4)
				{
				case 0:
					n = tensor.shape_[i];
					break;
				case 1:
					c = tensor.shape_[i];
					break;
				case 2:
					h = tensor.shape_[i];
					break;
				case 3:
					w = tensor.shape_[i];
					break;
				}
			}

			CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32, n, c, h, w));
		}
		else if (dim <= CUDNN_DIM_MAX)
		{
			std::vector<int> dims;
			for (size_t i = 0; i < dim; ++i)
				dims[i] = tensor.shape_[i];
			CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptorEx(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32, dim, dims.data()));
		}
		else
		{
			throw std::runtime_error("Invalid tensor dimension!");
		}

		return descriptor;
	}

	template <>
	inline tensor_descriptor convolution_extensions::make_tensor_descriptor(tensor<int8_t> const& tensor)
	{
		tensor_descriptor descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>(new cudnnTensorDescriptor_t(),
			[](cudnnTensorDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(descriptor.get()));

		auto dim = tensor_shape_dimension(tensor.shape_);
		if (dim < 5)
		{
			size_t n = 1, c = 1, h = 1, w = 1;

			for (size_t i = 0; i < dim; ++i)
			{
				switch (i % 4)
				{
				case 0:
					n = tensor.shape_[i];
					break;
				case 1:
					c = tensor.shape_[i];
					break;
				case 2:
					h = tensor.shape_[i];
					break;
				case 3:
					w = tensor.shape_[i];
					break;
				}
			}

			CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, n, c, h, w));
		}
		else if (dim <= CUDNN_DIM_MAX)
		{
			std::vector<int> dims;
			for (size_t i = 0; i < dim; ++i)
				dims[i] = tensor.shape_[i];
			CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptorEx(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, dim, dims.data()));
		}
		else
		{
			throw std::runtime_error("Invalid tensor dimension!");
		}

		return descriptor;
	}

	template <>
	inline tensor_descriptor convolution_extensions::make_tensor_descriptor(tensor<uint8_t> const& tensor)
	{
		tensor_descriptor descriptor = std::unique_ptr<cudnnTensorDescriptor_t, std::function<void(cudnnTensorDescriptor_t*)>>(new cudnnTensorDescriptor_t(),
			[](cudnnTensorDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(descriptor.get()));

		auto dim = tensor_shape_dimension(tensor.shape_);
		if (dim < 5)
		{
			size_t n = 1, c = 1, h = 1, w = 1;

			for (size_t i = 0; i < dim; ++i)
			{
				switch (i % 4)
				{
				case 0:
					n = tensor.shape_[i];
					break;
				case 1:
					c = tensor.shape_[i];
					break;
				case 2:
					h = tensor.shape_[i];
					break;
				case 3:
					w = tensor.shape_[i];
					break;
				}
			}

			CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_UINT8, n, c, h, w));
		}
		else if (dim <= CUDNN_DIM_MAX)
		{
			std::vector<int> dims;
			for (size_t i = 0; i < dim; ++i)
				dims[i] = tensor.shape_[i];
			CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptorEx(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_UINT8, dim, dims.data()));
		}
		else
		{
			throw std::runtime_error("Invalid tensor dimension!");
		}

		return descriptor;
	}

	inline activation_descriptor convolution_extensions::make_activation_descriptor(cudnnActivationMode_t activation_mode, cudnnNanPropagation_t nan_propagation, double coefficient)
	{
		activation_descriptor descriptor = std::unique_ptr<cudnnActivationDescriptor_t, std::function<void(cudnnActivationDescriptor_t*)>>(new cudnnActivationDescriptor_t(),
			[](cudnnActivationDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyActivationDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateActivationDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetActivationDescriptor(*descriptor, activation_mode, nan_propagation, coefficient));

		return descriptor;
	}

	inline pooling_descriptor convolution_extensions::make_2d_pooling_descriptor(cudnnPoolingMode_t pooling_mode,
		cudnnNanPropagation_t nan_propagation, int window_height, int window_width, int vertical_padding,
		int horizontal_padding, int vertical_stride, int horizontal_stride)
	{
		pooling_descriptor descriptor = std::unique_ptr<cudnnPoolingDescriptor_t, std::function<void(cudnnPoolingDescriptor_t*)>>(new cudnnPoolingDescriptor_t(),
			[](cudnnPoolingDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyPoolingDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreatePoolingDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetPooling2dDescriptor(
			*descriptor, pooling_mode, nan_propagation,
			window_height, window_width,
			vertical_padding, horizontal_padding,
			vertical_stride, horizontal_stride));

		return descriptor;
	}

	template<typename T>
	filter_descriptor convolution_extensions::make_4d_filter_descriptor(int k, int c, int h, int w)
	{
		filter_descriptor descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>(new cudnnFilterDescriptor_t(),
			[](cudnnFilterDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));

		return descriptor;
	}

	template<>
	inline filter_descriptor convolution_extensions::make_4d_filter_descriptor<half>(int k, int c, int h, int w)
	{
		filter_descriptor descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>(new cudnnFilterDescriptor_t(),
			[](cudnnFilterDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*descriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, k, c, h, w));

		return descriptor;
	}

	template<>
	inline filter_descriptor convolution_extensions::make_4d_filter_descriptor<int32_t>(int k, int c, int h, int w)
	{
		filter_descriptor descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>(new cudnnFilterDescriptor_t(),
			[](cudnnFilterDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*descriptor, CUDNN_DATA_INT32, CUDNN_TENSOR_NCHW, k, c, h, w));

		return descriptor;
	}

	template<>
	inline filter_descriptor convolution_extensions::make_4d_filter_descriptor<int8_t>(int k, int c, int h, int w)
	{
		filter_descriptor descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>(new cudnnFilterDescriptor_t(),
			[](cudnnFilterDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*descriptor, CUDNN_DATA_INT8, CUDNN_TENSOR_NCHW, k, c, h, w));

		return descriptor;
	}

	template<>
	inline filter_descriptor convolution_extensions::make_4d_filter_descriptor<uint8_t>(int k, int c, int h, int w)
	{
		filter_descriptor descriptor = std::unique_ptr<cudnnFilterDescriptor_t, std::function<void(cudnnFilterDescriptor_t*)>>(new cudnnFilterDescriptor_t(),
			[](cudnnFilterDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(*descriptor, CUDNN_DATA_UINT8, CUDNN_TENSOR_NCHW, k, c, h, w));

		return descriptor;
	}

	template <typename T>
	convolution_descriptor convolution_extensions::make_2d_convolution_descriptor(int padding_height, int padding_width,
		int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode)
	{
		convolution_descriptor descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>(new cudnnConvolutionDescriptor_t(),
			[](cudnnConvolutionDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*descriptor, padding_height, padding_width, u, v, dilation_height, dilation_width, convolution_mode, CUDNN_DATA_FLOAT));

		return descriptor;
	}

	template <>
	inline convolution_descriptor convolution_extensions::make_2d_convolution_descriptor<half>(int padding_height, int padding_width,
		int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode)
	{
		convolution_descriptor descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>(new cudnnConvolutionDescriptor_t(),
			[](cudnnConvolutionDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*descriptor, padding_height, padding_width, u, v, dilation_height, dilation_width, convolution_mode, CUDNN_DATA_HALF));

		return descriptor;
	}

	template <>
	inline convolution_descriptor convolution_extensions::make_2d_convolution_descriptor<int32_t>(int padding_height, int padding_width,
		int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode)
	{
		convolution_descriptor descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>(new cudnnConvolutionDescriptor_t(),
			[](cudnnConvolutionDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*descriptor, padding_height, padding_width, u, v, dilation_height, dilation_width, convolution_mode, CUDNN_DATA_INT32));

		return descriptor;
	}

	template <>
	inline convolution_descriptor convolution_extensions::make_2d_convolution_descriptor<int8_t>(int padding_height, int padding_width,
		int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode)
	{
		convolution_descriptor descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>(new cudnnConvolutionDescriptor_t(),
			[](cudnnConvolutionDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*descriptor, padding_height, padding_width, u, v, dilation_height, dilation_width, convolution_mode, CUDNN_DATA_INT8
		));

		return descriptor;
	}

	template <>
	inline convolution_descriptor convolution_extensions::make_2d_convolution_descriptor<uint8_t>(int padding_height, int padding_width,
		int u, int v, int dilation_height, int dilation_width, cudnnConvolutionMode_t convolution_mode)
	{
		convolution_descriptor descriptor = std::unique_ptr<cudnnConvolutionDescriptor_t, std::function<void(cudnnConvolutionDescriptor_t*)>>(new cudnnConvolutionDescriptor_t(),
			[](cudnnConvolutionDescriptor_t* obj)
		{
			CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*obj));
		});

		CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(descriptor.get()));
		CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(*descriptor, padding_height, padding_width, u, v, dilation_height, dilation_width, convolution_mode, CUDNN_DATA_UINT8));

		return descriptor;
	}

	inline workspace_memory convolution_extensions::make_workspace_memory(size_t size)
	{
		float* memory;
		CHECK_CUDA_ERROR(cudaMalloc(&memory, size));

		return std::unique_ptr<float[], std::function<void(float*)>>(memory,
			[](float* obj)
		{
			CHECK_CUDA_ERROR(cudaFree(obj));
		});;
	}

	inline tensor_shape convolution_extensions::get_conv2_forward_output_shape(
		convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor,
		filter_descriptor const& kernel_descriptor)
	{
		int n, c, h, w;
		CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(*convolution, *input_tensor_descriptor, *kernel_descriptor, &n, &c, &h, &w));
		return tensor_shape({ size_t(n),size_t(c),size_t(h),size_t(w) });
	}

	inline cudnnConvolutionFwdAlgo_t convolution_extensions::get_convolution_forward_algorithm(
		convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor,
		filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_tensor_descriptor)
	{
		cudnnConvolutionFwdAlgo_t algo;
		CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(tensor_lib_init::cudnn_handle(), *input_tensor_descriptor, *kernel_descriptor, *convolution, *output_tensor_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
		return algo;
	}

	inline size_t convolution_extensions::get_convolution_forward_workspace_size(
		convolution_descriptor const& convolution, tensor_descriptor const& input_tensor_descriptor,
		filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_tensor_descriptor,
		cudnnConvolutionFwdAlgo_t algo)
	{
		size_t size_in_bytes;
		CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(tensor_lib_init::cudnn_handle(), *input_tensor_descriptor, *kernel_descriptor, *convolution, *output_tensor_descriptor, algo, &size_in_bytes));
		return size_in_bytes;
	}

	inline cudnnConvolutionBwdDataAlgo_t convolution_extensions::get_convolution_backward_data_algorithm(
		filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_descriptor,
		convolution_descriptor const& convolution, tensor_descriptor const& input_descriptor)
	{
		cudnnConvolutionBwdDataAlgo_t algo;
		CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(tensor_lib_init::cudnn_handle(), *kernel_descriptor, *output_descriptor, *convolution, *input_descriptor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
		return algo;
	}

	inline size_t convolution_extensions::get_convolution_backward_data_workspace_size(
		filter_descriptor const& kernel_descriptor, tensor_descriptor const& output_descriptor,
		convolution_descriptor const& convolution, tensor_descriptor const& input_descriptor,
		cudnnConvolutionBwdDataAlgo_t algo)
	{
		size_t size_in_bytes;
		CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(tensor_lib_init::cudnn_handle(), *kernel_descriptor, *output_descriptor, *convolution, *input_descriptor, algo, &size_in_bytes));
		return size_in_bytes;
	}

	template <typename I, typename W, typename B, typename O>
	void convolution_extensions::convolution_forward(tensor_descriptor const& input_tensor_descriptor,
		tensor<I> const& input, filter_descriptor const& kernel_descriptor, tensor<W> const& weights,
		convolution_descriptor const& convolution, cudnnConvolutionFwdAlgo_t algo, workspace_memory const& memory,
		size_t workspace_size, tensor_descriptor const& output_tensor_descriptor, tensor<O>& output,
		tensor_descriptor const& bias_tensor_descriptor, tensor<B> const& bias)
	{
		float alpha = 1.0f, beta = 0.0f;
		CHECK_CUDNN_ERROR(cudnnConvolutionForward(
			tensor_lib_init::cudnn_handle(), &alpha, 
			*input_tensor_descriptor, input.data(), 
			*kernel_descriptor, weights.data(), 
			*convolution, algo, 
			memory.get(), workspace_size, &beta, 
			*output_tensor_descriptor, output.data()));

		CHECK_CUDNN_ERROR(cudnnAddTensor(
			tensor_lib_init::cudnn_handle(), &alpha, 
			*bias_tensor_descriptor, bias.data(), &alpha, 
			*output_tensor_descriptor, output.data()));
	}

	template <typename I, typename W, typename O>
	void convolution_extensions::convolution_backward_data(filter_descriptor const& kernel_descriptor,
		tensor<W> const& weights, tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output,
		convolution_descriptor const& convolution, cudnnConvolutionBwdDataAlgo_t algo, workspace_memory const& memory,
		size_t workspace_size, tensor_descriptor const& input_tensor_descriptor, tensor<I>& input)
	{
		float alpha = 1.0f, beta = 0.0f;

		CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(
			tensor_lib_init::cudnn_handle(), &alpha,
			*kernel_descriptor, weights.data(),
			*output_tensor_descriptor, output.data(),
			*convolution, algo,
			memory.get(), workspace_size, &beta,
			*input_tensor_descriptor, input.data()));
	}

	template <typename O, typename B>
	void convolution_extensions::convolution_backward_bias(tensor_descriptor const& output_descriptor,
		tensor<O> const& output, tensor_descriptor const& bias_tensor_descriptor, tensor<B>& grads_bias)
	{
		float alpha = 1.0f, beta = 0.0f;
		CHECK_CUDNN_ERROR(cudnnConvolutionBackwardBias(
			tensor_lib_init::cudnn_handle(), &alpha,
			*output_descriptor, output.data(),
			&beta, *bias_tensor_descriptor, grads_bias.data()));
	}

	inline cudnnConvolutionBwdFilterAlgo_t convolution_extensions::get_convolution_backward_filter_algorithm(
		tensor_descriptor const& input_tensor_descriptor, tensor_descriptor const& output_tensor_descriptor,
		convolution_descriptor const& convolution, filter_descriptor const& kernel_descriptor)
	{
		cudnnConvolutionBwdFilterAlgo_t algo;
		CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(tensor_lib_init::cudnn_handle(), *input_tensor_descriptor, *output_tensor_descriptor, *convolution, *kernel_descriptor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
		return algo;
	}

	inline size_t convolution_extensions::get_convolution_backward_filter_workspace_size(
		tensor_descriptor const& input_tensor_descriptor, tensor_descriptor const& output_tensor_descriptor,
		convolution_descriptor const& convolution, filter_descriptor const& kernel_descriptor,
		cudnnConvolutionBwdFilterAlgo_t algo)
	{
		size_t size_in_bytes;
		cudnnGetConvolutionBackwardFilterWorkspaceSize(tensor_lib_init::cudnn_handle(), *input_tensor_descriptor, *output_tensor_descriptor, *convolution, *kernel_descriptor, algo, &size_in_bytes);
		return size_in_bytes;
	}

	template <typename I, typename O, typename W>
	void convolution_extensions::convolution_backward_filter(tensor_descriptor const& input_tensor_descriptor,
		tensor<I> const& input, tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output,
		convolution_descriptor const& convolution, cudnnConvolutionBwdFilterAlgo_t algo, workspace_memory const& memory,
		size_t workspace_size, filter_descriptor const& kernel_descriptor, tensor<W>& grads_weight)
	{
		float alpha = 1.0f, beta = 0.0f;
		CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
			tensor_lib_init::cudnn_handle(), &alpha,
			*input_tensor_descriptor, input.data(),
			*output_tensor_descriptor, output.data(),
			*convolution, algo,
			memory.get(), workspace_size, &beta,
			*kernel_descriptor, grads_weight.data()));
	}

	template <typename I, typename O>
	void convolution_extensions::pooling_forward(pooling_descriptor const& pooling,
		tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input,
		tensor_descriptor const& output_tensor_descriptor, tensor<O>& output)
	{
		float alpha = 1.0f, beta = 0.0f;
		CHECK_CUDNN_ERROR(cudnnPoolingForward(tensor_lib_init::cudnn_handle(), *pooling, &alpha, *input_tensor_descriptor, input.data(), &beta, *output_tensor_descriptor, output.data()));
	}

	template <typename I, typename O, typename G, typename R>
	void convolution_extensions::pooling_backward(pooling_descriptor const& pooling,
		tensor_descriptor const& output_tensor_descriptor, tensor<O> const& output,
		tensor_descriptor const& grads_tensor_descriptor, tensor<G> const& grads,
		tensor_descriptor const& input_tensor_descriptor, tensor<I> const& input,
		tensor_descriptor const& result_descriptor, tensor<R>& result)
	{
		float alpha = 1.0f, beta = 0.0f;
		CHECK_CUDNN_ERROR(cudnnPoolingBackward(
			tensor_lib_init::cudnn_handle(), *pooling, &alpha,
			*output_tensor_descriptor, output.data(),
			*grads_tensor_descriptor, grads.data(),
			*input_tensor_descriptor, input.data(), &beta,
			*result_descriptor, result.data()));
	}
}
