#pragma once
#include "tensor.h"

namespace transprecision_floating_point
{
	namespace cuda_blas
	{

		using default_tensor = tensor<double>;

		inline void test_dot()
		{
			std::cout << "Testing tensor dot product..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto second = default_tensor::create<2, 3>({ 1,2,3,4,5,6 });
			auto exp_result = default_tensor::create<3, 3>({ 9,12,15,19,26,33,29,40,51 });

			auto first_q = default_tensor::create<3, 3>({ 1,2,3,4,5,6,7,8,9 });
			auto second_q = default_tensor::create<3, 3>({ 1,2,3,4,5,6,7,8,9 });
			auto exp_result_q = default_tensor::create<3, 3>({ 30,36,42,66,81,96,102,126,150 });

			auto result = dot(first, second);

			for (size_t i = 0; i < exp_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_result.shape().second; ++j)
				{
					if (result[{i, j}] != exp_result[{i, j}])
					{
						std::cout << "Dot test failed!" << std::endl;
						std::cout << result << "\n";
						return;
					}
				}
			}

			auto result_q = dot(first_q, second_q);

			for (size_t i = 0; i < exp_result_q.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_result_q.shape().second; ++j)
				{
					if (result_q[{i, j}] != exp_result_q[{i, j}])
					{
						std::cout << "Dot test failed!" << std::endl;
						return;
					}
				}
			}

			std::cout << "Dot test passed!" << std::endl;
		}


		inline void test_operators()
		{
			std::cout << "Testing tensor operators..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto second = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_add_result = default_tensor::create<3, 2>({ 2,4,6,8,10,12 });
			auto exp_sub_result = default_tensor::create<3, 2>({ 0,0,0,0,0,0 });
			auto exp_mul_result = default_tensor::create<3, 2>({ 1,4,9,16,25,36 });
			auto exp_div_result = default_tensor::create<3, 2>({ 1,1,1,1,1,1 });
			auto exp_single_result = default_tensor::create<3, 2>({ 1,5. / 3,7. / 3,3,11. / 3,13. / 3 });

			auto add_result = first + second;

			for (size_t i = 0; i < exp_add_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_add_result.shape().second; ++j)
				{
					if (add_result[{i, j}] != exp_add_result[{i, j}])
					{
						std::cout << "Operators test failed!" << std::endl;
						return;
					}
				}
			}

			auto sub_result = first - second;

			for (size_t i = 0; i < exp_sub_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_sub_result.shape().second; ++j)
				{
					if (sub_result[{i, j}] != exp_sub_result[{i, j}])
					{
						std::cout << "Operators test failed!" << std::endl;
						return;
					}
				}
			}

			auto mul_result = first * second;

			for (size_t i = 0; i < exp_mul_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_mul_result.shape().second; ++j)
				{
					if (mul_result[{i, j}] != exp_mul_result[{i, j}])
					{
						std::cout << "Operators test failed!" << std::endl;
						return;
					}
				}
			}

			auto div_result = first / second;

			for (size_t i = 0; i < exp_div_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_div_result.shape().second; ++j)
				{
					if (div_result[{i, j}] != exp_div_result[{i, j}])
					{
						std::cout << "Operators test failed!" << std::endl;
						return;
					}
				}
			}

			auto single_result = (((first + 1.) - 0.5)*2.) / 3.;

			for (size_t i = 0; i < exp_single_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_single_result.shape().second; ++j)
				{
					if (single_result[{i, j}] != exp_single_result[{i, j}])
					{
						std::cout << "Operators test failed!" << std::endl;
						return;
					}
				}
			}

			std::cout << "Operators test passed!" << std::endl;
		}

		inline void test_sum()
		{
			std::cout << "Testing tensor sum..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_sum_zero_result = default_tensor::create<1, 2>({ 9,12 });
			auto exp_sum_first_result = default_tensor::create<3, 1>({ 3,7,11 });
			auto exp_sum_result = 21.0;

			auto sum_zero_result = first.sum<0>();

			for (size_t i = 0; i < exp_sum_zero_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_sum_zero_result.shape().second; ++j)
				{
					if (sum_zero_result[{i, j}] != exp_sum_zero_result[{i, j}])
					{
						std::cout << "Sum test failed!" << std::endl;
						return;
					}
				}
			}

			auto sum_first_result = first.sum<1>();

			for (size_t i = 0; i < exp_sum_first_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_sum_first_result.shape().second; ++j)
				{
					if (sum_first_result[{i, j}] != exp_sum_first_result[{i, j}])
					{
						std::cout << "Sum test failed!" << std::endl;
						return;
					}
				}
			}

			auto sum_result = first.sum();
			if (sum_result != exp_sum_result)
			{
				std::cout << "Sum test failed!" << std::endl;
				return;
			}

			std::cout << "Sum test passed!" << std::endl;
		}

		inline void test_exp()
		{
			std::cout << "Testing tensor exp..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_result = default_tensor::create<3, 2>({ expf(1),expf(2),expf(3),expf(4),expf(5),expf(6) });

			auto result = exp(first);

			for (size_t i = 0; i < exp_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_result.shape().second; ++j)
				{
					if (result[{i, j}] != exp_result[{i, j}])
					{
						std::cout << "Exp test failed!" << std::endl;
						return;
					}
				}
			}

			std::cout << "Exp test passed!" << std::endl;
		}

		inline void test_apply()
		{
			std::cout << "Testing tensor apply..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_result = default_tensor::create<3, 2>({ 1,2,3,1,1,1 });

			auto result = apply(first, [] __device__ (double value) { return value > 3 ? 1 : value; });

			for (size_t i = 0; i < exp_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_result.shape().second; ++j)
				{
					if (result[{i, j}] != exp_result[{i, j}])
					{
						std::cout << "Apply test failed!" << std::endl;
						return;
					}
				}
			}

			std::cout << "Apply test passed!" << std::endl;
		}

		inline void test_max()
		{
			std::cout << "Testing tensor max..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_max_zero_result = default_tensor::create<1, 2>({ 5,6 });
			auto exp_max_first_result = default_tensor::create<3, 1>({ 2,4,6 });
			auto exp_max_result = 6.0;

			auto max_zero_result = first.max<0>();

			for (size_t i = 0; i < exp_max_zero_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_max_zero_result.shape().second; ++j)
				{
					if (max_zero_result[{i, j}] != exp_max_zero_result[{i, j}])
					{
						std::cout << "Max test failed!" << std::endl;
						return;
					}
				}
			}

			auto max_first_result = first.max<1>();

			for (size_t i = 0; i < exp_max_first_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_max_first_result.shape().second; ++j)
				{
					if (max_first_result[{i, j}] != exp_max_first_result[{i, j}])
					{
						std::cout << "Max test failed!" << std::endl;
						return;
					}
				}
			}

			auto max_result = first.max();
			if (max_result != exp_max_result)
			{
				std::cout << "Max test failed!" << std::endl;
				return;
			}

			std::cout << "Max test passed!" << std::endl;
		}

		inline void test_argmax()
		{
			std::cout << "Testing tensor argmax..." << std::endl;

			auto first = default_tensor::create<3, 2>({ 1,2,3,4,5,6 });
			auto exp_max_zero_result = tensor<size_t>::create<1, 2>({ 2,2 });
			auto exp_max_first_result = tensor<size_t>::create<3, 1>({ 1,1,1 });
			auto exp_max_result = tensor_index{ 2,1 };

			auto max_zero_result = first.argmax<0>();

			for (size_t i = 0; i < exp_max_zero_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_max_zero_result.shape().second; ++j)
				{
					if (max_zero_result[{i, j}] != exp_max_zero_result[{i, j}])
					{
						std::cout << "Argmax test failed!" << std::endl;
						return;
					}
				}
			}

			auto max_first_result = first.argmax<1>();

			for (size_t i = 0; i < exp_max_first_result.shape().first; ++i)
			{
				for (size_t j = 0; j < exp_max_first_result.shape().second; ++j)
				{
					if (max_first_result[{i, j}] != exp_max_first_result[{i, j}])
					{
						std::cout << "Argmax test failed!" << std::endl;
						return;
					}
				}
			}

			auto max_result = first.argmax();
			if (max_result != exp_max_result)
			{
				std::cout << "Argmax test failed!" << std::endl;
				return;
			}

			std::cout << "Argmax test passed!" << std::endl;
		}

		inline void test_tensor()
		{
			test_dot();
			test_operators();
			test_sum();
			test_exp();
			test_apply();
			test_max();
		}
	}
}