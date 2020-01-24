#include "pch.h"
#include <iostream>
#include "../simple_blas/tensor.h"
#include <random>
#include <chrono>
#include <fstream>
#include <functional>
#include "../simple_ml_lib/deep_model.h"
#include "../simple_ml_lib/fully_connected_layer.h"
#include "../simple_ml_lib/softmax_cross_entropy_with_logits_layer.h"
#include "../simple_ml_lib/sgd_optimizer.h"
#include "../simple_ml_lib/relu_layer.h"
#include "../flexfloat/flexfloat.hpp"
#include "../simple_ml_lib/gradient_checker.h"
#include "../simple_blas/tensor_tests.h"
#include "../simple_ml_lib/mean_squared_error.h"
#include "../simple_ml_lib/mnist_loader.h"
#include "../simple_ml_lib/adam_optimizer.h"

#define TEST_TENSOR

using float32 = flexfloat<8, 23>;
using float16 = flexfloat<5, 10>;
using float16alt = flexfloat<8, 7>;
using float8 = flexfloat<5, 2>;

using tensor32 = transprecision_floating_point::simple_blas::tensor<float32>;
using tensor16 = transprecision_floating_point::simple_blas::tensor<float16>;
using tensor16alt = transprecision_floating_point::simple_blas::tensor<float16alt>;
using tensor8 = transprecision_floating_point::simple_blas::tensor<float8>;
using tensor64 = transprecision_floating_point::simple_blas::tensor<double>;

template<typename Func>
void test_exec_time(Func fun, std::string_view test_name)
{
	std::cout << "Starting test: " << test_name.data() << std::endl;
	auto const start_time = omp_get_wtime();
	fun();
	auto const end_time = omp_get_wtime();;
	std::cout << "Time passed: " << end_time - start_time << "s" << std::endl;
}

void test_init()
{
	test_exec_time([]()
	{
		omp_set_num_threads(1);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float8>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "One thread - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(1);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float8>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "One thread - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(1);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float8>::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "One thread - 500 x 500 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(1);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float8>::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "One thread - 1000 x 1000 tensor");

	std::cout << std::endl;

	test_exec_time([]()
	{
		omp_set_num_threads(4);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Four threads - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(4);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Four threads - 500 x 500 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(4);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Four threads - 1000 x 1000 tensor");

	std::cout << std::endl;

	test_exec_time([]()
	{
		omp_set_num_threads(7);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Seven threads - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(7);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Seven threads - 500 x 500 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(7);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Seven threads - 1000 x 1000 tensor");

	std::cout << std::endl;

	test_exec_time([]()
	{
		omp_set_num_threads(8);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Eight threads - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(8);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Eight threads - 500 x 500 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(8);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Eight threads - 1000 x 1000 tensor");

	std::cout << std::endl;

	test_exec_time([]()
	{
		omp_set_num_threads(16);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Sixteen threads - 100 x 100 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(16);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Sixteen threads - 500 x 500 tensor");
	test_exec_time([]()
	{
		omp_set_num_threads(16);
		auto tensor = transprecision_floating_point::simple_blas::tensor<float32>::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	}, "Sixteen threads - 1000 x 1000 tensor");
}

void test_basic_op()
{
	auto tensor100 = tensor8::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor500 = tensor8::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor1000 = tensor8::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));

	for (auto element : { 1,4,7,8 })
	{
		for (auto i = 0; i < 3; ++i)
		{
			test_exec_time([element, &tensor100, &tensor500, &tensor1000, i]()
			{
				omp_set_num_threads(element);
				tensor8 tensor;
				switch (i)
				{
				case 0:
					tensor = tensor100 + tensor100;
					break;
				case 1:
					tensor = tensor500 + tensor500;
					break;
				case 2:
					tensor = tensor1000 + tensor1000;
					break;
				}
			}, "Thread count: " + std::to_string(element) + " - tensor size: " + std::to_string(i == 0 ? 100 : i == 1 ? 500 : 1000));
		}

		std::cout << std::endl;
	}
}

void test_mat_dot_op()
{
	auto tensor100 = tensor8::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor500 = tensor8::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor1000 = tensor8::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));

	for (auto element : { 4,7,8 })
	{
		for (auto i = 0; i < 3; ++i)
		{
			test_exec_time([element, &tensor100, &tensor500, &tensor1000, i]()
			{
				omp_set_num_threads(element);
				tensor8 tensor;
				switch (i)
				{
				case 0:
					tensor = dot(tensor100, tensor100);
					break;
				case 1:
					tensor = dot(tensor500, tensor500);
					break;
				case 2:
					tensor = dot(tensor1000, tensor1000);
					break;
				}
			}, "Thread count: " + std::to_string(element) + " - tensor size: " + std::to_string(i == 0 ? 100 : i == 1 ? 500 : 1000));
		}

		std::cout << std::endl;
	}
}

void test_sum()
{
	auto tensor100 = tensor8::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor500 = tensor8::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor1000 = tensor8::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));

	for (auto element : { 1,4,7,8 })
	{
		for (auto i = 0; i < 3; ++i)
		{
			test_exec_time([element, &tensor100, &tensor500, &tensor1000, i]()
			{
				omp_set_num_threads(element);
				float8 sum = 0;
				switch (i)
				{
				case 0:
					sum = tensor100.sum();
					break;
				case 1:
					sum = tensor500.sum();
					break;
				case 2:
					sum = tensor1000.sum();
					break;
				}
			}, "Thread count: " + std::to_string(element) + " - tensor size: " + std::to_string(i == 0 ? 100 : i == 1 ? 500 : 1000));
		}

		std::cout << std::endl;
	}
}

void test_log()
{
	auto tensor100 = tensor8::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor500 = tensor8::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor1000 = tensor8::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));

	for (auto element : { 1,4,7,8 })
	{
		for (auto i = 0; i < 3; ++i)
		{
			test_exec_time([element, &tensor100, &tensor500, &tensor1000, i]()
			{
				omp_set_num_threads(element);
				tensor8 tensor;
				switch (i)
				{
				case 0:
					tensor = tensor100.log();
					break;
				case 1:
					tensor = tensor500.log();
					break;
				case 2:
					tensor = tensor1000.log();
					break;
				}
			}, "Thread count: " + std::to_string(element) + " - tensor size: " + std::to_string(i == 0 ? 100 : i == 1 ? 500 : 1000));
		}

		std::cout << std::endl;
	}
}

void test_exp()
{
	auto tensor100 = tensor8::create_random<100, 100, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor500 = tensor8::create_random<500, 500, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));
	auto tensor1000 = tensor8::create_random<1000, 1000, std::uniform_real_distribution<float>>(std::uniform_real_distribution<float>(0, 1));

	for (auto element : { 1,4,7,8 })
	{
		for (auto i = 0; i < 3; ++i)
		{
			test_exec_time([element, &tensor100, &tensor500, &tensor1000, i]()
			{
				omp_set_num_threads(element);
				tensor8 tensor;
				switch (i)
				{
				case 0:
					tensor = tensor100.exp();
					break;
				case 1:
					tensor = tensor500.exp();
					break;
				case 2:
					tensor = tensor1000.exp();
					break;
				}
			}, "Thread count: " + std::to_string(element) + " - tensor size: " + std::to_string(i == 0 ? 100 : i == 1 ? 500 : 1000));
		}

		std::cout << std::endl;
	}
}

template<typename Type>
std::pair<transprecision_floating_point::simple_blas::tensor<Type>, transprecision_floating_point::simple_blas::tensor<Type>> generate_data(size_t size)
{
	auto x = transprecision_floating_point::simple_blas::tensor<Type>::template create_random<std::uniform_real_distribution<float>>(size, 2, std::uniform_real_distribution<float>(0, 1));
	transprecision_floating_point::simple_blas::tensor<Type> y({ size, 2 });

	for (size_t i = 0; i < size; ++i)
	{
		if (x[{i, 0}] > 0.5)
			y[{i, 1}] = 1;
		else
			y[{i, 0}] = 1;

		/*if (transprecision_floating_point::simple_blas::thread_safe_rand(std::uniform_real_distribution<float>(0, 1)) < 0.1)
			y[{i, 0}] = 1 - y[{i, 0}];*/
	}

	return { x,y };
}

void test_transpose()
{
	for (int i = 0; i < 5; ++i)
	{
		auto x = transprecision_floating_point::simple_blas::tensor<float>::create_random<std::uniform_real_distribution<float>>(1000, 2, std::uniform_real_distribution<float>(0, 1));
		test_exec_time([&x]()
		{
			for (auto i = 0; i < 10; ++i)
				x.transpose();

		}, "Transpose test");
	}
}

template<typename Type>
void check_gradient()
{
	transprecision_floating_point::simple_ml_lib::gradient_checker<Type> g_chk;
	auto dist = std::uniform_real_distribution<float>(0, 1);

	{
		std::cout << "ReLU:\n";
		auto x = transprecision_floating_point::simple_blas::tensor<Type>::create_random(20, 40, dist);
		auto grad_out = transprecision_floating_point::simple_blas::tensor<Type>::create_random(20, 40, dist);
		transprecision_floating_point::simple_ml_lib::relu_layer<Type> relu("relu1");
		std::cout << "Check grad wrt input\n";
		g_chk.check_grad_inputs(relu, x, grad_out);
		std::cout << std::endl;
	}

	{
		std::cout << "FC:\n";
		auto x = transprecision_floating_point::simple_blas::tensor<Type>::create_random(20, 40, dist);
		auto grad_out = transprecision_floating_point::simple_blas::tensor<Type>::create_random(20, 30, dist);
		transprecision_floating_point::simple_ml_lib::fully_connected_layer<Type> fc(40, 30,
			transprecision_floating_point::simple_ml_lib::uniform_real_random_initializer<Type>,
			transprecision_floating_point::simple_ml_lib::zeros_initializer<Type>, "fc1");
		std::cout << "Check grad wrt input\n";
		g_chk.check_grad_inputs(fc, x, grad_out);
		std::cout << "Check grad wrt params\n";
		g_chk.check_grad_params(fc, x, fc.weights_, fc.bias_, grad_out);
		std::cout << std::endl;
	}

	{
		std::cout << "SoftmaxCrossEntropyWithLogits:\n";
		auto x = transprecision_floating_point::simple_blas::tensor<Type>::create_random(50, 20, dist);
		transprecision_floating_point::simple_blas::tensor<Type> y({ 50, 20 });
		for (size_t i = 0; i < 50; ++i)
			y[{i, 0}] = 1;
		transprecision_floating_point::simple_ml_lib::softmax_cross_entropy_with_logits<Type> loss(1e-10);
		auto grad_x_num = g_chk.eval_numerical_gradient([&loss, &y](auto const& x) { return loss.forward(x, y); }, x, Type(1), Type(1e-5));
		auto out = loss.forward(x, y);
		auto grad_x = loss.backward_inputs(x, y);
		std::cout << "Relative error = " << transprecision_floating_point::simple_ml_lib::gradient_checker<Type>::rel_error(grad_x_num, grad_x);
		std::cout << std::endl;
	}
}

#ifndef TEST_TENSOR
int main()
{
	try
	{
		using fp = float8;
#define EPSILON 1e-5
#define EPOCH_COUNT 100
#define SOFTMAX
#ifdef MSE
#define LOSS_FUNCTION transprecision_floating_point::simple_ml_lib::mean_squared_error<fp>
		transprecision_floating_point::simple_ml_lib::mean_squared_error<fp> loss;
#elif defined SOFTMAX
		transprecision_floating_point::simple_ml_lib::softmax_cross_entropy_with_logits<fp> loss(EPSILON);
#define LOSS_FUNCTION transprecision_floating_point::simple_ml_lib::softmax_cross_entropy_with_logits<fp>
#endif

		transprecision_floating_point::simple_ml_lib::mnist_dataset<fp> data("mnist");

		transprecision_floating_point::simple_ml_lib::deep_model<fp, LOSS_FUNCTION> model(loss);
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

		transprecision_floating_point::simple_ml_lib::adam_optimizer<fp> opt(1, 0.9,0.91, 1e-5);
		for (auto i = 0; i < EPOCH_COUNT; ++i)
		{
			auto const logits = opt.train_epoch(model, data.train_images, data.train_labels);

			if (i % 1 == 0)
			{
				std::cout << "Iteration " << i + 1 << " loss " << model.loss(logits, data.train_labels) << "\n";
				std::cout << "Iteration " << i + 1 << " accuracy " << model.accuracy(logits, data.train_labels) << "\n" << std::endl;
			}
		}
		//check_gradient<double>();
	}
	catch (std::exception const& exc)
	{
		std::cerr << exc.what();
	}
}

#else

int main(int argc, char* argv[])
{
	float8 a = 4.f;

	std::cout << a << "\n";
}

#endif

