#pragma once
#include "tensor.h"
#include "thrust_helpers.h"

namespace transprecision_floating_point
{
	struct dnn_extensions
	{
		template<typename T, typename U, typename R>
		static R cross_entropy_loss(tensor<T> const& probs, tensor<U> const& y, float epsilon = 1e-6);
		template<typename T, typename U, typename R>
		static R cross_entropy_loss(tensor<T> const& probs, tensor<U> const& y, tensor_precision const& precision, float epsilon = 1e-6);
		template<typename T, typename U, typename O, typename P, typename Q, typename R, typename S, typename V, typename Z>
		static void update_weights_adam(
			tensor<T>& weights, tensor<U>& bias, tensor<O> const& grad_weights, tensor<P> const& grad_bias,
			tensor<Q>& weight_s, tensor<R>& bias_s, tensor<S>& weight_r, tensor<V>& bias_r, Z learning_rate, Z decay_s, Z decay_r, size_t t, Z delta);
		template<typename T, typename U, typename Rng>
		static std::pair<std::vector<tensor<T>>, std::vector<tensor<U>>> permutate_and_split_data(tensor<T>& x, tensor<U>& y, Rng&& rng, size_t batch_size);
	};

	template <typename T, typename U, typename R>
	R dnn_extensions::cross_entropy_loss(tensor<T> const& probs, tensor<U> const& y, float epsilon)
	{
		auto n = probs.shape_.front();
		auto total_size = tensor_shape_total_size(probs.shape_);
		thrust::device_ptr<T> probs_ptr(probs.data());
		thrust::device_ptr<U> y_ptr(y.data());
		auto fun = [epsilon] __device__(thrust::tuple<T, U>& data) { return arithmetic_operators<float, U, R>::mul(logf(float(thrust::get<0>(data) == 0 ? epsilon : thrust::get<0>(data))), thrust::get<1>(data)); };
		auto sum = [] __device__(R first, R second) { return arithmetic_operators<R, R, R>::add(first, second); };

		auto first = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(probs_ptr, y_ptr)), fun);

		return R(-1.f * thrust::reduce(first, first + total_size, R(0), sum) / n);
	}

	template <typename T, typename U, typename R>
	R dnn_extensions::cross_entropy_loss(tensor<T> const& probs, tensor<U> const& y, tensor_precision const& precision, float epsilon)
	{
		auto n = probs.shape_.front();
		auto total_size = tensor_shape_total_size(probs.shape_);
		thrust::device_ptr<T> probs_ptr(probs.data());
		thrust::device_ptr<U> y_ptr(y.data());
		auto fun = [epsilon] __device__(thrust::tuple<T, U>& data) { return arithmetic_operators<float, U, R>::mul(logf(float(thrust::get<0>(data) == 0 ? epsilon : thrust::get<0>(data))), thrust::get<1>(data)); };
		auto sum = [] __device__(R first, R second) { return arithmetic_operators<R, R, R>::add(first, second); };

		auto first = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(probs_ptr, y_ptr)), fun);

		return sanitize(precision.exp_bits, precision.frac_bits, R(-1.f * thrust::reduce(first, first + total_size, R(0), sum) / n));
	}

	template <typename T, typename U, typename O, typename P, typename Q, typename R, typename S, typename V, typename Z>
	void dnn_extensions::update_weights_adam(tensor<T>& weights, tensor<U>& bias, tensor<O> const& grad_weights,
		tensor<P> const& grad_bias, tensor<Q>& weight_s, tensor<R>& bias_s, tensor<S>& weight_r, tensor<V>& bias_r,
		Z learning_rate, Z decay_s, Z decay_r, size_t t, Z delta)
	{
		auto weights_total_size = tensor_shape_total_size(weights.shape_);
		auto bias_total_size = tensor_shape_total_size(bias.shape_);

		auto weights_ = weights.data();
		auto bias_ = bias.data();
		auto grad_weights_ = grad_weights.data();
		auto grad_bias_ = grad_bias.data();
		auto weight_s_ = weight_s.data();
		auto bias_s_ = bias_s.data();
		auto weight_r_ = weight_r.data();
		auto bias_r_ = bias_r.data();

		auto should_update_weights = weights.should_sanitize();
		auto should_update_bias = bias.should_sanitize();
		auto exp_bits_weights = weights.precision_.exp_bits;
		auto frac_bits_weights = weights.precision_.frac_bits;
		auto exp_bits_bias = bias.precision_.exp_bits;
		auto frac_bits_bias = bias.precision_.frac_bits;

		auto update_fun = [
			weights_, bias_, grad_weights_,
				grad_bias_, weight_s_,
				bias_s_, weight_r_, bias_r_,
				learning_rate, decay_s,
				decay_r, t, delta,
				weights_total_size, bias_total_size,
				should_update_weights, should_update_bias,
				exp_bits_weights, frac_bits_weights,
				exp_bits_bias, frac_bits_bias
		]__device__(size_t index)
			{
				if (index < weights_total_size)
				{
					weight_s_[index] = decay_s * weight_s_[index] + (1 - decay_s) * grad_weights_[index];
					weight_r_[index] = decay_r * weight_r_[index] + (1 - decay_r) * (grad_weights_[index] * grad_weights_[index]);
					
					weight_s_[index] /= (1 - Q(powf(float(decay_s), t)));
					weight_r_[index] /= (1 - S(powf(float(decay_r), t)));
					
					auto delta_weight = -learning_rate * (weight_s_[index] / (sqrtf(float(weight_r_[index])) + delta));

					if (should_update_bias)
						weights_[index] = sanitize(exp_bits_weights, frac_bits_weights, weights_[index] + delta_weight);
					else
						weights_[index] += delta_weight;
				}

				if (index < bias_total_size)
				{

					bias_s_[index] = decay_s * bias_s_[index] + (1 - decay_s) * grad_bias_[index];
					bias_r_[index] = decay_r * bias_r_[index] + (1 - decay_r) * (grad_bias_[index] * grad_bias_[index]);

					bias_s_[index] /= (1 - R(powf(float(decay_s), t)));
					bias_r_[index] /= (1 - V(powf(float(decay_r), t)));

					auto delta_bias = -learning_rate * (bias_s_[index] / (sqrtf(float(bias_r_[index])) + delta));

					if(should_update_bias)
						bias_[index] = sanitize(exp_bits_bias,frac_bits_bias, bias_[index] + delta_bias);
					else
						bias_[index] += delta_bias;

				}
			};

			thrust::for_each(
				thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(0) + std::max(weights_total_size, bias_total_size), update_fun);
	}


	template<typename T>
	struct batch_permutator
	{
		__host__ __device__ batch_permutator(
			size_t stride, 
			size_t* indices, T* data): 
			stride_(stride), 
			indices_(indices), 
			data_(data) {  }

		__host__ __device__ T operator()(size_t index)
		{
			auto const i = index / stride_;
			return data_[indices_[i] * stride_ + (index % stride_)];
		}

		size_t stride_;
		size_t* indices_;
		T* data_;
	};

	template <typename T, typename U, typename Rng>
	std::pair<std::vector<tensor<T>>, std::vector<tensor<U>>> dnn_extensions::permutate_and_split_data(tensor<T>& x, tensor<U>& y, Rng&& rng, size_t batch_size)
	{
		auto const n = x.shape_.front();
		if (n != y.shape_.front())
			throw std::runtime_error("Invalid shapes!");

		auto const stride_x = tensor_shape_total_size(x.shape_) / n;
		auto const stride_y = tensor_shape_total_size(y.shape_) / n;

		std::vector<size_t> indices(n);
		thrust::sequence(thrust::host, indices.begin(), indices.end());
		std::shuffle(indices.begin(), indices.end(), rng);

		thrust::device_vector<size_t> indices_device(indices.begin(), indices.end());

		auto num_of_batches = n / batch_size + (n % batch_size != 0 ? 1 : 0);
		std::vector<tensor<T>> x_batches(num_of_batches);
		std::vector<tensor<U>> y_batches(num_of_batches);

		thrust::device_ptr<T> x_ptr(x.data());
		thrust::device_ptr<U> y_ptr(y.data());

		auto shuffle_x_first = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), batch_permutator<T>(stride_x, indices_device.data().get(), x.data()));
		auto shuffle_y_first = thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), batch_permutator<U>(stride_y, indices_device.data().get(), y.data()));

		for (size_t i = 0; i < num_of_batches; ++i)
		{
			auto new_x_shape = x.shape_;
			new_x_shape[0] = (i == num_of_batches - 1 && (n % batch_size != 0)) ? n % batch_size : batch_size;
			tensor<T> batch_x(new_x_shape, x.precision_);
			thrust::device_ptr<T> batch_x_ptr(batch_x.data());
			auto batch_shuffle_x_first = shuffle_x_first + i * batch_size * stride_x;
			thrust::copy(batch_shuffle_x_first, batch_shuffle_x_first + new_x_shape[0] * stride_x, batch_x_ptr);
			x_batches[i] = std::move(batch_x);

			auto new_y_shape = y.shape_;
			new_y_shape[0] = (i == num_of_batches - 1 && (n % batch_size != 0)) ? n % batch_size : batch_size;
			tensor<U> batch_y(new_y_shape, y.precision_);
			thrust::device_ptr<U> batch_y_ptr(batch_y.data());
			auto batch_shuffle_y_first = shuffle_y_first + i * batch_size * stride_y;
			thrust::copy(batch_shuffle_y_first, batch_shuffle_y_first + new_y_shape[0] * stride_y, batch_y_ptr);
			y_batches[i] = std::move(batch_y);
		}

		return std::pair<std::vector<tensor<T>>, std::vector<tensor<U>>>(std::move(x_batches), std::move(y_batches));
	}
}
