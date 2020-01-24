
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "tensor_tests.h"
#include <fstream>
#include <chrono>
#include "distributions.h"
#include "../flexfloat_cuda/flexfloat_cuda.h"
using float32 = transprecision_floating_point::cuda::flexfloat_cuda<8, 23>;
using float16 = transprecision_floating_point::cuda::flexfloat_cuda<5, 10>;
using float16alt = transprecision_floating_point::cuda::flexfloat_cuda<8, 7>;
using float8 = transprecision_floating_point::cuda::flexfloat_cuda<5, 2>;

int main()
{
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	try
	{
		/*double avg_time = 0;
		auto t = transprecision_floating_point::cuda_blas::tensor<float8>::create_random(500, 100, transprecision_floating_point::cuda_blas::normal_distribution());
		for (auto i = 0; i < 100; ++i)
		{

			auto const start_time = std::chrono::high_resolution_clock::now();
			auto sum = t.max<1>();
			auto const end_time = std::chrono::high_resolution_clock::now();
			auto time = std::chrono::duration<float, std::chrono::milliseconds::period>(end_time - start_time).count();
			std::cout << "Time passed: " << time << "ms" << std::endl;
			avg_time += time;
		}

		std::cout << "Avg time passed: " << avg_time / 100 << "ms" << std::endl;*/

		/*auto t = transprecision_floating_point::cuda_blas::tensor<float8>::create_random(100, 100, transprecision_floating_point::cuda_blas::normal_distribution());
		transprecision_floating_point::cuda_blas::tensor<float8> t_r({ 100,1 }, float8(1.0));
		std::fstream file("output.txt", std::fstream::out);
		file << t << "\n\n";
		file << produce(t, t_r, [] __device__(float8 first, float8 second) { return first + second; });*/
		/*auto t = transprecision_floating_point::cuda_blas::tensor<float>::create_random(10, 5, transprecision_floating_point::cuda_blas::normal_distribution());
		auto t_1 = transprecision_floating_point::cuda_blas::tensor<float>::create_random(10, 5, transprecision_floating_point::cuda_blas::normal_distribution());
		
		std::fstream file("output.txt", std::fstream::out);
		file << t << "\n\n";
		file << transpose(t) << "\n\n";
		file << t_1 << "\n\n";
		file << transpose(t_1);*/
		transprecision_floating_point::cuda_blas::test_tensor();
	}
	catch (std::exception const& exc)
	{
		std::cerr << exc.what() << "\n";
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//transprecision_floating_point::cuda_blas::test_tensor();
	return 0;
}
