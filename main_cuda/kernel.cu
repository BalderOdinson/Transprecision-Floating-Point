
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "../flexfloat_cuda/flexfloat_cuda.h"
#include "../simple_ml_lib_cuda/softmax_cross_entropy_with_logits_layer.h"
#include "../simple_ml_lib_cuda/mnist_loader.h"
#include "../simple_ml_lib_cuda/deep_model.h"
#include "../simple_ml_lib_cuda/fully_connected_layer.h"
#include "../simple_ml_lib_cuda/relu_layer.h"
#include "../simple_ml_lib_cuda/adam_optimizer.h"
#include "../simple_ml_lib_cuda/sgd_optimizer.h"
#include <chrono>
#define USE_CUDA

using float8 = transprecision_floating_point::cuda::flexfloat_cuda<5, 2>;
using float16 = transprecision_floating_point::cuda::flexfloat_cuda<5, 10>;
using float16alt = transprecision_floating_point::cuda::flexfloat_cuda<8, 7>;
using float32 = transprecision_floating_point::cuda::flexfloat_cuda<8, 23>;

int main()
{
	try
	{
		using fp = float;
#define EPSILON 1e-8
#define EPOCH_COUNT 30
#define BATCH_SIZE 1000
#define TRAIN_SIZE 60000
#define SOFTMAX
#ifdef MSE
#define LOSS_FUNCTION transprecision_floating_point::simple_ml_lib::mean_squared_error<fp>
		transprecision_floating_point::simple_ml_lib::mean_squared_error<fp> loss_fun;
#elif defined SOFTMAX
		transprecision_floating_point::simple_ml_lib::softmax_cross_entropy_with_logits<fp> loss_fun(EPSILON);
#define LOSS_FUNCTION transprecision_floating_point::simple_ml_lib::softmax_cross_entropy_with_logits<fp>
#endif

		transprecision_floating_point::simple_ml_lib::mnist_dataset<fp> data("mnist");

		transprecision_floating_point::simple_ml_lib::deep_model<fp, LOSS_FUNCTION> model(loss_fun);
		model += std::make_unique<transprecision_floating_point::simple_ml_lib::fully_connected_layer<fp>>(784, 100,
			transprecision_floating_point::simple_ml_lib::variance_normal_initializer<fp>(data.train_images.shape().first),
			transprecision_floating_point::simple_ml_lib::zeros_initializer<fp>, "fc1");
		model += std::make_unique<transprecision_floating_point::simple_ml_lib::relu_layer<fp>>("relu1");
		model += std::make_unique<transprecision_floating_point::simple_ml_lib::fully_connected_layer<fp>>(100, 100,
			transprecision_floating_point::simple_ml_lib::variance_normal_initializer<fp>(data.train_images.shape().first),
			transprecision_floating_point::simple_ml_lib::zeros_initializer<fp>, "fc2");
		model += std::make_unique<transprecision_floating_point::simple_ml_lib::relu_layer<fp>>("relu2");
		model += std::make_unique<transprecision_floating_point::simple_ml_lib::fully_connected_layer<fp>>(100, 10,
			transprecision_floating_point::simple_ml_lib::variance_normal_initializer<fp>(data.train_images.shape().first),
			transprecision_floating_point::simple_ml_lib::zeros_initializer<fp>, "logits");

		transprecision_floating_point::simple_ml_lib::adam_optimizer<fp> opt(0.005, 0.9, 0.999, 1e-8);
		/*auto lr = 0.01;
		transprecision_floating_point::simple_ml_lib::sgd_optimizer<fp> opt(lr);*/
		fp accuracy = 0;
		size_t i = 0;
		while (accuracy < 0.01)
		{
			//lr *= 0.999;
			/*auto perm_data = permutate(data.train_images, data.train_labels, size_t(i));
			auto train_image_batches = perm_data.first.split(TRAIN_SIZE / BATCH_SIZE);
			auto train_labels_batches = perm_data.second.split(TRAIN_SIZE / BATCH_SIZE);
			for (size_t j = 0; j < train_image_batches.size(); ++j)
			{
				auto const logits = opt.train_epoch(model, train_image_batches[j], train_labels_batches[j]);
				if (i % 1 == 0)
				{
					std::cout << "Iteration " << i + 1 << ", batch " + std::to_string(j) + "/" + std::to_string(train_image_batches.size()) + ", loss " << model.loss(logits, train_labels_batches[j]) << "\n";
					std::cout << "Iteration " << i + 1 << ", batch " + std::to_string(j) + "/" + std::to_string(train_image_batches.size()) + ", accuracy " << model.accuracy(logits, train_labels_batches[j]) << "\n";
				}

			}

			auto const test_logits = model.feed_forward(data.train_images);
			std::cout << "\nIteration " << i + 1 << " loss " << model.loss(test_logits, data.train_labels) << "\n";
			std::cout << "Iteration " << i + 1 << " accuracy " << model.accuracy(test_logits, data.train_labels) << "\n\n";*/

			auto const logits = opt.train_epoch(model, data.train_images, data.train_labels);
			accuracy = model.accuracy(logits, data.train_labels);
			std::cout << "Iteration " << i + 1 << " loss " << model.loss(logits, data.train_labels) << "\n";
			std::cout << "Iteration " << i + 1 << " accuracy " << accuracy << "\n" << std::endl;

			++i;
			/*if(i == EPOCH_COUNT - 1)
			{
				auto const test_logits = model.feed_forward(data.test_images);
				std::cout << "Test loss " << model.loss(test_logits, data.test_labels) << "\n";
				std::cout << "Test accuracy " << model.accuracy(test_logits, data.test_labels) << "\n";
			}*/
		}
		auto const test_logits = model.feed_forward(data.test_images);
		std::cout << "Test loss " << model.loss(test_logits, data.test_labels) << "\n";
		std::cout << "Test accuracy " << model.accuracy(test_logits, data.test_labels) << "\n";
		//check_gradient<double>();
	}
	catch (std::exception const& exc)
	{
		std::cerr << exc.what();
	}
}

// Helper function for using CUDA to add vectors in parallel.