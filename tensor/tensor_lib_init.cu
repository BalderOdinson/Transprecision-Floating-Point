#include "tensor_lib_init.h"


namespace transprecision_floating_point
{
	void tensor_lib_init::init(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle)
	{
		if (cublas_handle)
		{
			cublas_handle_() = cublas_handle;
			return;
		}

		if(cudnn_handle)
		{
			cudnn_handle_() = cudnn_handle;
			return;
		}

		CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle_()));
		CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle_()));
	}

	cublasHandle_t tensor_lib_init::cublas_handle()
	{
		return cublas_handle_();
	}

	cudnnHandle_t tensor_lib_init::cudnn_handle()
	{
		return cudnn_handle_();
	}

	void tensor_lib_init::destroy()
	{
		cublasDestroy(cublas_handle_());
		cudnnDestroy(cudnn_handle_());
	}

	cublasHandle_t& tensor_lib_init::cublas_handle_()
	{
		static cublasHandle_t handle = nullptr;
		return handle;
	}

	cudnnHandle_t& tensor_lib_init::cudnn_handle_()
	{
		static cudnnHandle_t handle = nullptr;
		return handle;
	}
}
