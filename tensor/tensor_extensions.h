#pragma once
#include "tensor.h"

namespace transprecision_floating_point
{
	struct tensor_extensions
	{
		template<typename T, typename U, typename R>
		static void gemm(float alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& b, bool transpose_b, float beta, tensor<R>& c);
		template<typename T, typename U, typename R>
		static void gemm_ex(float alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& b, bool transpose_b, float beta, bool transpose_c, tensor<R>& c);
		template<typename T, typename U>
		static void axpy(U alpha, tensor<T> const& x, tensor<U>& y);
		template<typename T, typename U, typename R>
		static void geam(R alpha, tensor<T> const& a, bool transpose_a, R beta, tensor<U> const& b, bool transpose_b, tensor<R>& c);
		template<typename T, typename U, typename R>
		static void gemv(R alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& x, bool transpose_x, R beta, bool transpose_y, tensor<R>& y);

		template<typename T, typename R = T>
		static tensor<R> transpose(tensor<T> const& a);
		template<typename T, typename R = T>
		static tensor<R> transpose(tensor<T> const& a, tensor_precision const& precision);
		template<typename T, typename U, typename R>
		static tensor<R> dot(tensor<T> const& a, tensor<U> const& b);
	};

	template <typename T, typename U, typename R>
	void tensor_extensions::gemm(float alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& b,
		bool transpose_b, float beta, tensor<R>& c)
	{
		auto const total_size_first = tensor_shape_total_size(a.shape_);
		auto const total_size_second = tensor_shape_total_size(b.shape_);

		auto rows_a = a.shape_.front();
		auto cols_a = total_size_first / rows_a;
		auto k_a = cols_a;
		auto n = rows_a;

		auto rows_b = b.shape_.front();
		auto cols_b = total_size_second / rows_b;
		auto m = cols_b;
		auto k_b = rows_b;

		if (transpose_a)
			std::swap(n, k_a);
		if (transpose_b)
			std::swap(m, k_b);

		tensor<float> a_cast;
		tensor<float> b_cast;
		if (!std::is_same<T, float>::value)
			a_cast = a;
		if (!std::is_same<U, float>::value)
			b_cast = b;

		tensor<float> result;
		if (beta == 0)
			result = tensor<float>(tensor_shape{ n, m });
		else
			result = std::move(c);

		CHECK_CUBLAS_ERROR(cublasSgemm(
			tensor_lib_init::cublas_handle(),
			transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
			transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
			m, n, k_b, &alpha,
			std::is_same<T, float>::value ? b.data() : b_cast.data(), cols_b,
			std::is_same<U, float>::value ? a.data() : a_cast.data(), cols_a,
			&beta, result.data(), m));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		c = tensor<R>(std::move(result), c.precision_);
	}

	template <typename T, typename U, typename R>
	void tensor_extensions::gemm_ex(float alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& b,
		bool transpose_b, float beta, bool transpose_c, tensor<R>& c)
	{
		if (beta == 0)
		{
			tensor_extensions::gemm(alpha, a, transpose_a, b, transpose_b, beta, c);
			return;
		}

		auto const total_size_first = tensor_shape_total_size(a.shape_);
		auto const total_size_second = tensor_shape_total_size(b.shape_);
		auto const total_size_c = tensor_shape_total_size(c.shape_);

		auto rows_a = a.shape_.front();
		auto cols_a = total_size_first / rows_a;
		auto k_a = cols_a;
		auto n = rows_a;

		auto rows_b = b.shape_.front();
		auto cols_b = total_size_second / rows_b;
		auto m = cols_b;
		auto k_b = rows_b;

		auto rows_c = c.shape_.front();
		auto cols_c = total_size_c / rows_c;

		if (transpose_a)
			std::swap(n, k_a);
		if (transpose_b)
			std::swap(m, k_b);

		tensor<float> a_cast;
		tensor<float> b_cast;
		if (!std::is_same<T, float>::value)
			a_cast = a;
		if (!std::is_same<U, float>::value)
			b_cast = b;

		auto c_shape = c.shape_;
		tensor_shape c_new_shape{ n, m };
		auto c_new_total_size = tensor_shape_total_size(c_new_shape);
		if (transpose_c)
		{
			tensor_shape new_shape_c(c_shape.begin() + 1, c_shape.end());
			new_shape_c.push_back(c_shape.front());
			c_shape = new_shape_c;
		}

		tensor<float> result;
		if (c_shape == c_new_shape)
		{
			if (transpose_c)
			{
				result = tensor<float>(c_new_shape);
				thrust::device_ptr<U> dev_ptr_c(c.data_.get());
				thrust::device_ptr<float> dev_ptr_r(result.data_.get());
				auto c_t = thrust::make_permutation_iterator(dev_ptr_c, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(rows_c, cols_c)));
				thrust::transform(c_t, c_t + c_new_total_size, dev_ptr_r, thrust::identity<float>());
				
			}
			else
			{
				result = std::move(c);
			}
		}
		else
		{
			if (transpose_c)
			{
				result = tensor<float>(c_new_shape);
				thrust::device_ptr<U> dev_ptr_c(c.data_.get());
				thrust::device_ptr<float> dev_ptr_r(result.data_.get());
				auto c_t = thrust::make_permutation_iterator(dev_ptr_c, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(rows_c, cols_c)));
				auto c_b = transprecision_floating_point::make_broadcast_iterator(c_t, c_new_shape, c_shape);
				thrust::transform(c_b, c_b + c_new_total_size, dev_ptr_r, thrust::identity<float>());
			}
			else
			{
				result = tensor<float>(c_new_shape);
				thrust::device_ptr<U> dev_ptr_c(c.data_.get());
				thrust::device_ptr<float> dev_ptr_r(result.data_.get());
				auto c_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_c, c_new_shape, c_shape);
				thrust::transform(c_b, c_b + c_new_total_size, dev_ptr_r, thrust::identity<float>());
			}
		}

		CHECK_CUBLAS_ERROR(cublasSgemm(
			tensor_lib_init::cublas_handle(),
			transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
			transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
			m, n, k_b, &alpha,
			std::is_same<T, float>::value ? b.data() : b_cast.data(), cols_b,
			std::is_same<U, float>::value ? a.data() : a_cast.data(), cols_a,
			&beta, result.data(), m));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		c = tensor<R>(std::move(result), c.precision_);

	}

	template <typename T, typename U>
	void tensor_extensions::axpy(U alpha, tensor<T> const& x, tensor<U>& y)
	{
		auto fun = [alpha] __device__(T first, U second) { return arithmetic_operators<T, U, U>::fma(alpha, first, second); };
		auto const total_size = tensor_shape_total_size(x.shape_);
		thrust::device_ptr<T> dev_ptr_x(x.data_.get());
		thrust::device_ptr<U> dev_ptr_y(y.data_.get());

		if (x.shape_ != y.shape_)
			throw std::runtime_error("Invalid shapes - " + to_string(x.shape_) + " and " + to_string(y.shape_) + "!");

		if (y.should_sanitize())
			thrust::transform(dev_ptr_x, dev_ptr_x + total_size, dev_ptr_y, dev_ptr_y, binary_sanitize<decltype(fun), T, U, U>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
		else
			thrust::transform(dev_ptr_x, dev_ptr_x + total_size, dev_ptr_y, dev_ptr_y, fun);
	}

	template <typename T, typename U, typename R>
	void tensor_extensions::geam(R alpha, tensor<T> const& a, bool transpose_a, R beta, tensor<U> const& b, bool transpose_b,
		tensor<R>& c)
	{
		thrust::device_ptr<T> dev_ptr_a(a.data_.get());
		thrust::device_ptr<U> dev_ptr_b(b.data_.get());

		auto a_shape = a.shape_;
		auto b_shape = b.shape_;

		auto a_total_size = tensor_shape_total_size(a_shape);
		auto b_total_size = tensor_shape_total_size(b_shape);

		auto a_n = a_shape.front();
		auto a_m = a_total_size / a_n;
		auto b_n = b_shape.front();
		auto b_m = b_total_size / b_n;

		auto fun = [alpha, beta] __device__(T first, U second)
		{
			return  arithmetic_operators<T, R, R>::fma(alpha, first, arithmetic_operators<U, R, R>::mul(second, beta));
		};

		if (transpose_a && transpose_b)
		{
			auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
			auto b_t = thrust::make_permutation_iterator(dev_ptr_b, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(b_n, b_m)));

			tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
			new_shape_a.push_back(a_shape.front());
			a_shape = new_shape_a;
			std::swap(a_n, a_m);

			tensor_shape new_shape_b(b_shape.begin() + 1, b_shape.end());
			new_shape_b.push_back(b_shape.front());
			b_shape = new_shape_b;
			std::swap(b_n, b_m);

			if (tensor_shape_total_size(c.shape_) != a_total_size)
				c = tensor<R>(a_shape, c.precision_);

			thrust::device_ptr<R> dev_ptr_c(c.data_.get());

			if (a_shape == b_shape)
			{
				if (c.should_sanitize())
					thrust::transform(a_t, a_t + a_total_size, b_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits));
				else
					thrust::transform(a_t, a_t + a_total_size, b_t, dev_ptr_c, fun);
			}
			else
			{
				if (c.should_sanitize())
					transprecision_floating_point::broadcast_transform(a_t, b_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits), a_shape, b_shape);
				else
					transprecision_floating_point::broadcast_transform(a_t, b_t, dev_ptr_c, fun, a_shape, b_shape);
			}
		}

		else if (transpose_a)
		{
			auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));

			tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
			new_shape_a.push_back(a_shape.front());
			a_shape = new_shape_a;
			std::swap(a_n, a_m);

			if (tensor_shape_total_size(c.shape_) != a_total_size)
				c = tensor<R>(a_shape, c.precision_);

			thrust::device_ptr<R> dev_ptr_c(c.data_.get());


			if (a_shape == b_shape)
			{
				if (c.should_sanitize())
					thrust::transform(a_t, a_t + a_total_size, dev_ptr_b, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits));
				else
					thrust::transform(a_t, a_t + a_total_size, dev_ptr_b, dev_ptr_c, fun);
			}
			else
			{
				if (c.should_sanitize())
					transprecision_floating_point::broadcast_transform(a_t, dev_ptr_b, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits), a_shape, b_shape);
				else
					transprecision_floating_point::broadcast_transform(a_t, dev_ptr_b, dev_ptr_c, fun, a_shape, b_shape);
			}
		}

		else if (transpose_b)
		{
			auto b_t = thrust::make_permutation_iterator(dev_ptr_b, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(b_n, b_m)));

			tensor_shape new_shape_b(b_shape.begin() + 1, b_shape.end());
			new_shape_b.push_back(b_shape.front());
			b_shape = new_shape_b;
			std::swap(b_n, b_m);

			if (tensor_shape_total_size(c.shape_) != a_total_size)
				c = tensor<R>(a_shape, c.precision_);

			thrust::device_ptr<R> dev_ptr_c(c.data_.get());

			if (a_shape == b_shape)
			{
				if (c.should_sanitize())
					thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, b_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits));
				else
					thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, b_t, dev_ptr_c, fun);
			}
			else
			{
				if (c.should_sanitize())
					transprecision_floating_point::broadcast_transform(dev_ptr_a, b_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits), a_shape, b_shape);
				else
					transprecision_floating_point::broadcast_transform(dev_ptr_a, b_t, dev_ptr_c, fun, a_shape, b_shape);
			}
		}

		else
		{
			if (tensor_shape_total_size(c.shape_) != a_total_size)
				c = tensor<R>(a_shape, c.precision_);

			thrust::device_ptr<R> dev_ptr_c(c.data_.get());

			if (a_shape == b_shape)
			{
				if (c.should_sanitize())
					thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, dev_ptr_b, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits));
				else
					thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, dev_ptr_b, dev_ptr_c, fun);
			}
			else
			{
				if (c.should_sanitize())
					transprecision_floating_point::broadcast_transform(dev_ptr_a, dev_ptr_b, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, c.precision_.exp_bits, c.precision_.frac_bits), a_shape, b_shape);
				else
					transprecision_floating_point::broadcast_transform(dev_ptr_a, dev_ptr_b, dev_ptr_c, fun, a_shape, b_shape);
			}
		}
	}

	template <typename T, typename U, typename R>
	void tensor_extensions::gemv(R alpha, tensor<T> const& a, bool transpose_a, tensor<U> const& x, bool transpose_x,
		R beta, bool transpose_y, tensor<R>& y)
	{
		thrust::device_ptr<T> dev_ptr_a(a.data_.get());
		thrust::device_ptr<U> dev_ptr_x(x.data_.get());

		auto a_shape = a.shape_;
		auto x_shape = x.shape_;

		auto a_total_size = tensor_shape_total_size(a_shape);
		auto x_total_size = tensor_shape_total_size(x_shape);

		auto a_n = a_shape.front();
		auto a_m = a_total_size / a_n;
		auto x_n = x_shape.front();
		auto x_m = x_total_size / x_n;

		if (beta == R(0))
		{
			auto fun = [alpha] __device__(T first, U second)
			{
				return  arithmetic_operators<T, U, R>::mul(arithmetic_operators<R, T, T>::mul(alpha, first), second);
			};

			if (transpose_a && transpose_x)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));

				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				if (tensor_shape_total_size(y.shape_) != a_total_size)
					y = tensor<R>(a_shape, y.precision_);

				thrust::device_ptr<R> dev_ptr_c(y.data_.get());

				if (a_shape == x_shape)
				{
					if (y.should_sanitize())
						thrust::transform(a_t, a_t + a_total_size, x_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(a_t, a_t + a_total_size, x_t, dev_ptr_c, fun);
				}
				else
				{
					if (y.should_sanitize())
						transprecision_floating_point::broadcast_transform(a_t, x_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits), a_shape, x_shape);
					else
						transprecision_floating_point::broadcast_transform(a_t, x_t, dev_ptr_c, fun, a_shape, x_shape);
				}
			}

			else if (transpose_a)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));

				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				if (tensor_shape_total_size(y.shape_) != a_total_size)
					y = tensor<R>(a_shape, y.precision_);

				thrust::device_ptr<R> dev_ptr_c(y.data_.get());


				if (a_shape == x_shape)
				{
					if (y.should_sanitize())
						thrust::transform(a_t, a_t + a_total_size, dev_ptr_x, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(a_t, a_t + a_total_size, dev_ptr_x, dev_ptr_c, fun);
				}
				else
				{
					if (y.should_sanitize())
						transprecision_floating_point::broadcast_transform(a_t, dev_ptr_x, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits), a_shape, x_shape);
					else
						transprecision_floating_point::broadcast_transform(a_t, dev_ptr_x, dev_ptr_c, fun, a_shape, x_shape);
				}
			}

			else if (transpose_x)
			{
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				if (tensor_shape_total_size(y.shape_) != a_total_size)
					y = tensor<R>(a_shape, y.precision_);

				thrust::device_ptr<R> dev_ptr_c(y.data_.get());

				if (a_shape == x_shape)
				{
					if (y.should_sanitize())
						thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, x_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, x_t, dev_ptr_c, fun);
				}
				else
				{
					if (y.should_sanitize())
						transprecision_floating_point::broadcast_transform(dev_ptr_a, x_t, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits), a_shape, x_shape);
					else
						transprecision_floating_point::broadcast_transform(dev_ptr_a, x_t, dev_ptr_c, fun, a_shape, x_shape);
				}
			}

			else
			{
				if (tensor_shape_total_size(y.shape_) != a_total_size)
					y = tensor<R>(a_shape, y.precision_);

				thrust::device_ptr<R> dev_ptr_c(y.data_.get());

				if (a_shape == x_shape)
				{
					if (y.should_sanitize())
						thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, dev_ptr_x, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(dev_ptr_a, dev_ptr_a + a_total_size, dev_ptr_x, dev_ptr_c, fun);
				}
				else
				{
					if (y.should_sanitize())
						transprecision_floating_point::broadcast_transform(dev_ptr_a, dev_ptr_x, dev_ptr_c, binary_sanitize<decltype(fun), T, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits), a_shape, x_shape);
					else
						transprecision_floating_point::broadcast_transform(dev_ptr_a, dev_ptr_x, dev_ptr_c, fun, a_shape, x_shape);
				}
			}
		}
		else
		{
			auto y_shape = y.shape_;
			auto y_total_size = tensor_shape_total_size(y_shape);
			auto y_n = y_shape.front();
			auto y_m = y_total_size / y_n;

			thrust::device_ptr<R> dev_ptr_y(y.data_.get());

			auto fun = [alpha, beta] __device__(auto first, U second)
			{
				return arithmetic_operators<T, U, R>::fma(arithmetic_operators<R, T, T>::mul(alpha, thrust::get<0>(first)), second, arithmetic_operators<R, R, R>::mul(beta, thrust::get<1>(first)));
			};

			if (transpose_a && transpose_x && transpose_y)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));
				auto y_t = thrust::make_permutation_iterator(dev_ptr_y, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(y_n, y_m)));
				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				tensor_shape new_shape_y(y_shape.begin() + 1, y_shape.end());
				new_shape_y.push_back(y_shape.front());
				y_shape = new_shape_y;
				std::swap(y_n, y_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_a && transpose_x)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));

				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_a && transpose_y)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
				auto y_t = thrust::make_permutation_iterator(dev_ptr_y, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(y_n, y_m)));
				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				tensor_shape new_shape_y(y_shape.begin() + 1, y_shape.end());
				new_shape_y.push_back(y_shape.front());
				y_shape = new_shape_y;
				std::swap(y_n, y_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_a)
			{
				auto a_t = thrust::make_permutation_iterator(dev_ptr_a, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(a_n, a_m)));
				tensor_shape new_shape_a(a_shape.begin() + 1, a_shape.end());
				new_shape_a.push_back(a_shape.front());
				a_shape = new_shape_a;
				std::swap(a_n, a_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(a_t, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_x && transpose_y)
			{
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));
				auto y_t = thrust::make_permutation_iterator(dev_ptr_y, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(y_n, y_m)));

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				tensor_shape new_shape_y(y_shape.begin() + 1, y_shape.end());
				new_shape_y.push_back(y_shape.front());
				y_shape = new_shape_y;
				std::swap(y_n, y_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_x)
			{
				auto x_t = thrust::make_permutation_iterator(dev_ptr_x, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(x_n, x_m)));

				tensor_shape new_shape_x(x_shape.begin() + 1, x_shape.end());
				new_shape_x.push_back(x_shape.front());
				x_shape = new_shape_x;
				std::swap(x_n, x_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(x_t, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_t, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else if (transpose_y)
			{
				auto y_t = thrust::make_permutation_iterator(dev_ptr_y, thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), linear_index_transpose(y_n, y_m)));

				tensor_shape new_shape_y(y_shape.begin() + 1, y_shape.end());
				new_shape_y.push_back(y_shape.front());
				y_shape = new_shape_y;
				std::swap(y_n, y_m);

				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(y_t, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_t));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
			}
			else
			{
				tensor<R> result(a_shape, y.precision_);
				thrust::device_ptr<R> dev_ptr_r(result.data_.get());

				if (a_shape != x_shape && a_shape != y_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);

					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != x_shape)
				{
					auto x_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_x, a_shape, x_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, x_b, dev_ptr_r, fun);

					y = std::move(result);
				}
				else if (a_shape != y_shape)
				{
					auto y_b = transprecision_floating_point::make_broadcast_iterator(dev_ptr_y, a_shape, y_shape);
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, y_b));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
				else
				{
					auto ax = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_a, dev_ptr_y));

					if (y.should_sanitize())
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, binary_sanitize<decltype(fun), thrust::tuple<T, R>, U, R>(fun, y.precision_.exp_bits, y.precision_.frac_bits));
					else
						thrust::transform(ax, ax + a_total_size, dev_ptr_x, dev_ptr_r, fun);

					y = std::move(result);
				}
			}

		}
	}

	template <typename T, typename R>
	tensor<R> tensor_extensions::transpose(tensor<T> const& a)
	{
		return tensor<R>(std::move(~a));
	}

	template <typename T, typename R>
	tensor<R> tensor_extensions::transpose(tensor<T> const& a, tensor_precision const& precision)
	{
		return tensor<R>(std::move(~a), precision);
	}

	template <typename T, typename U, typename R>
	tensor<R> tensor_extensions::dot(tensor<T> const& a, tensor<U> const& b)
	{
		return tensor<R>(std::move(a.dot(b)));
	}
}
