#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include "fully_connected_layer.h"
#include "cublas_v2.h"
#include <chrono>
#include "relu_layer.h"
#include "softmax_cross_entropy_with_logits.h"
#include "sgd_optimizer.h"
#include "mnist_loader.h"
#include "adam_optimizer.h"
#include "conv2d_layer.h"
#include "max_pool_layer.h"
#include "flatten_layer.h"
#include <thread>
#include "shared_workspace_memory.h"

#define NEW_LINE "\n"
#define DOUBLE_NEW_LINE "\n\n"


template<typename T>
transprecision_floating_point::tensor<T> uniform_real_random_initializer(transprecision_floating_point::tensor_shape shape)
{
	return transprecision_floating_point::tensor<T>::random(shape, transprecision_floating_point::uniform_distribution());
}

template<typename T>
transprecision_floating_point::tensor<T> zeros_initializer(transprecision_floating_point::tensor_shape shape)
{
	return transprecision_floating_point::tensor<T>(shape, T(0));
}

template<typename T>
struct xavier_normal_initializer
{
	xavier_normal_initializer(T inputs, T outputs) :inputs_(std::move(inputs)), outputs_(std::move(outputs)) {  }
	transprecision_floating_point::tensor<T> operator()(transprecision_floating_point::tensor_shape shape)
	{

		return transprecision_floating_point::tensor<T>::random(shape, transprecision_floating_point::xavier_distribution(inputs_, outputs_));
	}

private:
	T inputs_;
	T outputs_;
};

int main()
{
	using fp = float;

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	transprecision_floating_point::tensor_lib_init::init();

	cublasHandle_t handle = transprecision_floating_point::tensor_lib_init::cublas_handle();

	try
	{
		transprecision_floating_point::random_engine::set_seed(100);
		transprecision_floating_point::mnist_dataset<float> dataset("mnist");
		dataset.train_images.reshape({ 60000,1,28,28 });
		dataset.test_images.reshape({ 10000,1,28,28 });

		transprecision_floating_point::softmax_cross_entropy_with_logits softmax;
		transprecision_floating_point::deep_model<transprecision_floating_point::softmax_cross_entropy_with_logits> model(softmax);
		model += std::make_unique<transprecision_floating_point::conv2d_layer<fp>>(
			transprecision_floating_point::conv2d_layer<fp>::make_conv2d_layer(
				1, 16, 5, 5, 1, 1, 
				xavier_normal_initializer<float>(25, 16), zeros_initializer<float>, "conv1", 
				FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8));
		model += std::make_unique<transprecision_floating_point::max_pool_layer>(
			transprecision_floating_point::max_pool_layer::make_max_pool_layer(
				2, 2, 2, 2, "pool1", FLOAT8));
		model += std::make_unique<transprecision_floating_point::conv2d_layer<fp>>(
			transprecision_floating_point::conv2d_layer<fp>::make_conv2d_layer(
				16, 32, 5, 5, 1, 1, 
				xavier_normal_initializer<float>(400, 32), zeros_initializer<float>, "conv2",
				FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8));
		model += std::make_unique<transprecision_floating_point::max_pool_layer>(
			transprecision_floating_point::max_pool_layer::make_max_pool_layer(
				2, 2, 2, 2, "pool2", FLOAT8));
		model += std::make_unique<transprecision_floating_point::flatten_layer>(
			transprecision_floating_point::flatten_layer::make_flatten_layer("flatten"));
		model += std::make_unique<transprecision_floating_point::fully_connected_layer>(
			transprecision_floating_point::fully_connected_layer::make_fully_connected_layer(
				512, 512, xavier_normal_initializer<float>(512, 512), zeros_initializer<float>, "fc1",
				FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8));
		model += std::make_unique<transprecision_floating_point::relu_layer>(
			transprecision_floating_point::relu_layer::make_relu_layer("relu1", FLOAT8));
		model += std::make_unique<transprecision_floating_point::fully_connected_layer>(
			transprecision_floating_point::fully_connected_layer::make_fully_connected_layer(
				512, 10, xavier_normal_initializer<float>(512, 10), zeros_initializer<float>, "logits",
				FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8));
		{
			auto lr = 1.0;
			transprecision_floating_point::adam_optimizer<float> optimizer(0.003, 0.995, 0.999, 1e-8);
			std::mt19937 rng(100);

			for (size_t i = 0; i < 5; ++i)
			{
				/*transprecision_floating_point::sgd_optimizer optimizer(lr);
				lr *= 0.9;
				lr = std::max(lr, 1e-3);*/
				{
					auto batch = transprecision_floating_point::dnn_extensions::permutate_and_split_data(dataset.train_images, dataset.train_labels, rng, 1000);
					auto batch_count = batch.first.size();

					for (size_t j = 0; j < batch_count; ++j)
					{
						std::cout << "----------------------------------------------------" << NEW_LINE;
						auto logits = optimizer.train_epoch(model, batch.first[j], batch.second[j]);
						auto accuracy = model.accuracy(logits, batch.second[j]);
						auto loss = softmax.forward(logits, batch.second[j]);
						std::cout << "Batch loss " << loss << " Batch " << std::to_string(j + 1) + "\\" + std::to_string(batch_count) << " Epoch " << i + 1 << NEW_LINE;
						std::cout << "Batch accuracy " << accuracy << " Batch " << std::to_string(j + 1) + "\\" + std::to_string(batch_count) << " Epoch " << i + 1 << NEW_LINE;
						std::cout << "----------------------------------------------------" << NEW_LINE;
					}
				}
				/*auto logits = optimizer.train_epoch(model, dataset.train_images, dataset.train_labels);
				auto accuracy = model.accuracy(logits, dataset.train_labels);
				auto loss = softmax.forward(logits, dataset.train_labels);
				std::cout << "Loss " << loss << " Epoch " << i + 1 << NEW_LINE;
				std::cout << "Accuracy " << accuracy << " Epoch " << i + 1 << DOUBLE_NEW_LINE;*/
				std::cout << NEW_LINE;
				auto logits = model.feed_forward(dataset.train_images);
				auto accuracy = model.accuracy(logits, dataset.train_labels);
				auto loss = softmax.forward(logits, dataset.train_labels);
				std::cout << "Loss " << loss << " Epoch " << i + 1 << NEW_LINE;
				std::cout << "Accuracy " << accuracy << " Epoch " << i + 1 << DOUBLE_NEW_LINE;
			}
		}

		auto logits = model.feed_forward(dataset.test_images);
		auto accuracy = model.accuracy(logits, dataset.test_labels);
		auto loss = softmax.forward(logits, dataset.test_labels);
		std::cout << "Test set loss " << loss << NEW_LINE;
		std::cout << "Test set accuracy " << accuracy;

		std::fstream conv1_file("conv1.txt", std::fstream::out);
		std::fstream conv2_file("conv2.txt", std::fstream::out);

		conv1_file << reinterpret_cast<transprecision_floating_point::conv2d_layer<fp>*>(model["conv1"].get())->get_weights() << DOUBLE_NEW_LINE;
		conv1_file << reinterpret_cast<transprecision_floating_point::conv2d_layer<fp>*>(model["conv1"].get())->get_bias() << DOUBLE_NEW_LINE;
		conv2_file << reinterpret_cast<transprecision_floating_point::conv2d_layer<fp>*>(model["conv2"].get())->get_weights() << DOUBLE_NEW_LINE;
		conv2_file << reinterpret_cast<transprecision_floating_point::conv2d_layer<fp>*>(model["conv2"].get())->get_bias() << DOUBLE_NEW_LINE;

		transprecision_floating_point::shared_workspace_memory::release();
	}
	catch (std::exception const& exc)
	{
		std::cerr << exc.what() << NEW_LINE;
	}

	transprecision_floating_point::tensor_lib_init::destroy();
	transprecision_floating_point::random_engine::destroy();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}