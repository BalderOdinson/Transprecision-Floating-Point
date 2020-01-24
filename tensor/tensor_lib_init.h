#ifndef TENSOR_LIB_INIT_H
#define TENSOR_LIB_INIT_H
#include <cublas_v2.h>
#include "cuda_error_defines.h"
#include <stdexcept>
#include <memory>
#include <functional>
#include <cudnn.h>

namespace transprecision_floating_point
{
	struct tensor_lib_init
	{
		static void init(cublasHandle_t cublas_handle = nullptr, cudnnHandle_t cudnn_handle = nullptr);
		static cublasHandle_t cublas_handle();
		static cudnnHandle_t cudnn_handle();
		static void destroy();

	private:
		static cublasHandle_t& cublas_handle_();
		static cudnnHandle_t& cudnn_handle_();
	};
}

#endif
