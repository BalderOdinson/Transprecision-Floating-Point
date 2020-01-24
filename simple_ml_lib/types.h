#pragma once

#ifdef USE_CUDA
#include "../cuda_blas/tensor.h"
using namespace transprecision_floating_point::cuda_blas;
#define DEVICE_FUNCTION __device__ 
#else
#include "../simple_blas/tensor.h"
using namespace transprecision_floating_point::simple_blas;
#define DEVICE_FUNCTION
#endif