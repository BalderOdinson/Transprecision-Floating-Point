#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include "tensor_lib.h"
#include "cublas_v2.h"
#include "curand.h"
#include <chrono>
#include "convolution_extensions.h"

#define NEW_LINE "\n"
#define DOUBLE_NEW_LINE "\n\n"

template<typename T>
void diff_checker(transprecision_floating_point::tensor<T> const& first, transprecision_floating_point::tensor<T> const& second)
{
	size_t diff = 0;
	for (size_t i = 0; i < first.shape()[0]; ++i)
	{
		for (size_t j = 0; j < first.shape()[1]; ++j)
		{
			float f = first[transprecision_floating_point::tensor_shape({ i,j })];
			float s = second[transprecision_floating_point::tensor_shape({ i,j })];
			if (fabs(f - s) > 0.00001f)
			{
				std::cout << "Diff at index: (" << std::to_string(i) + "," + std::to_string(j) << ") Values: (" << std::to_string(f) + "," + std::to_string(s) << ")\n";
				++diff;
			}
		}
	}

	std::cout << "Differences found: " << diff << "\n";
}

//int main()
//{
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	auto cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		return 1;
//	}
//
//	transprecision_floating_point::tensor_lib_init::init();
//
//	cublasHandle_t handle = transprecision_floating_point::tensor_lib_init::cublas_handle();
//
//	try
//	{
//		transprecision_floating_point::random_engine::set_seed(100);
//		auto a = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 1000, 1000 }, transprecision_floating_point::normal_distribution());
//		auto b = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 1000, 1000 }, transprecision_floating_point::normal_distribution());
//		//transprecision_floating_point::tensor<float> c(transprecision_floating_point::tensor_shape{ 60000,750 });
//
//		double avg_time_blas = 0;
//		double avg_time = 0;
//		for (auto i = 0; i < 100; ++i)
//		{
//			{
//				auto const start_time = std::chrono::high_resolution_clock::now();
//				{
//					auto result = b.argmax(1);
//				}
//
//				auto const end_time = std::chrono::high_resolution_clock::now();
//				auto time = std::chrono::duration<float, std::chrono::milliseconds::period>(end_time - start_time).count();
//				std::cout << "Time passed: " <<  time << "ms" << std::endl;
//				avg_time_blas += time;
//			}
//
//			{
//				/*auto const start_time = std::chrono::high_resolution_clock::now();
//				{
//					auto result = a.argmax(1);
//				}
//				auto const end_time = std::chrono::high_resolution_clock::now();
//				auto time = std::chrono::duration<float, std::chrono::milliseconds::period>(end_time - start_time).count();
//				std::cout << "Time passed: " << time << "ms" << std::endl;
//				avg_time += time;*/
//			}
//		}
//
//		std::cout << "Avg time 60000: " << avg_time_blas / 100 << NEW_LINE;
//		//std::cout << "Avg time 1000: " << avg_time / 100 << NEW_LINE;
//
//		//std::fstream file("output.txt", std::fstream::out);
//		//std::fstream ex_file("expected_output.txt", std::fstream::out);
//		//auto a = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 4,2 }, transprecision_floating_point::normal_distribution());
//		//auto b = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 2,3 }, transprecision_floating_point::normal_distribution());;
//		//auto y = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 1,4 }, transprecision_floating_point::normal_distribution());;
//		////auto result = a.dot(~a);
//
//		//file << a << DOUBLE_NEW_LINE << a.argmax(1);
//
//		//transprecision_floating_point::tensor_extensions::gemm_ex(1, a, false, b, false, 1,true, c);
//		////transprecision_floating_point::tensor_extensions::gemv(1.f, a, true, b, true, 1.f, true, c);
//		//
//		///*file <<
//		//	~a << DOUBLE_NEW_LINE <<
//		//	~b << DOUBLE_NEW_LINE <<
//		//	~y << DOUBLE_NEW_LINE <<
//		//	~a * ~b << DOUBLE_NEW_LINE <<
//		//	c << DOUBLE_NEW_LINE;*/
//
//		//auto res = (a.dot(b)) + ~y;
//
//		//diff_checker(res, c);
//
//		/*file << c;
//		ex_file << (~a * ~b) + ~y;*/
//
//		/*transprecision_floating_point::tensor_extensions::axpy(-2.f, c, c);
//		file << DOUBLE_NEW_LINE << c;*/
//		/*file << a << DOUBLE_NEW_LINE << a.sum() << DOUBLE_NEW_LINE << a.argmax(0, false) << DOUBLE_NEW_LINE << a.argmin(1, false) << DOUBLE_NEW_LINE << a.argmax(2, false);*/
//
//		/*auto a = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 2,2,2,2 }, transprecision_floating_point::normal_distribution());
//		auto b = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 2,2,1 }, transprecision_floating_point::normal_distribution());
//		auto c = transprecision_floating_point::tensor<float>::random(transprecision_floating_point::tensor_shape{ 1,2,2 }, transprecision_floating_point::normal_distribution());
//		auto d = 5.f;
//
//		std::fstream file("output.txt", std::fstream::out);
//		file << "MATRIX A -----------------------------------------------------" << NEW_LINE << a << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A+A ---------------------------------------------------" << NEW_LINE << a + a << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A -----------------------------------------------------" << NEW_LINE << a << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX B -----------------------------------------------------" << NEW_LINE << b << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A+B ---------------------------------------------------" << NEW_LINE << a + b << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A -----------------------------------------------------" << NEW_LINE << a << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX C -----------------------------------------------------" << NEW_LINE << c << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A+C ---------------------------------------------------" << NEW_LINE << a + c << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A -----------------------------------------------------" << NEW_LINE << a << NEW_LINE << "--------------------------------------------------------------" << DOUBLE_NEW_LINE
//			<< "MATRIX A+D ---------------------------------------------------" << NEW_LINE << a + d << NEW_LINE << "--------------------------------------------------------------";*/
//	}
//	catch (std::exception const& exc)
//	{
//		std::cerr << exc.what() << NEW_LINE;
//	}
//
//	transprecision_floating_point::tensor_lib_init::destroy();
//
//	// cudaDeviceReset must be called before exiting in order for profiling and
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//
//
//
//	return 0;
//}
