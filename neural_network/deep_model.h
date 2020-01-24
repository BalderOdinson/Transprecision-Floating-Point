#pragma once
#include "layer.h"
#include <unordered_map>

namespace transprecision_floating_point
{
	using model_layer = std::unique_ptr<layer>;

	template<typename Loss>
	struct deep_model
	{
		explicit deep_model(Loss loss);
		deep_model& operator+=(model_layer&& layer);
		deep_model& operator-=(std::string layer_name);
		model_layer& operator[](std::string name);
		model_layer& operator[](size_t index);
		tensor<float> forward_pass(tensor<float> const& x);
		tensor<float> feed_forward(tensor<float> const& x);
		std::vector<backward_grads<float,float,float,float>> backward_pass(tensor<float> const& x, tensor<float> const& y_oh);
		float loss(tensor<float> const& logits, tensor<float> const& y_oh);
		double accuracy(tensor<float> const& logits, tensor<float> const& y_oh);
		double accuracy_from_data(tensor<float> const& inputs, tensor<float> const& y_oh);

		tensor<size_t> evaluate(tensor<float> const& x);
	private:
		Loss loss_;
		std::unordered_map<std::string, size_t> layer_dict_;
		std::unordered_map<size_t, model_layer> layers_;
		size_t depth_{ 0 };
	};

	template <typename Loss>
	deep_model<Loss>::deep_model(Loss loss) : loss_(loss)
	{
	}

	template <typename Loss>
	deep_model<Loss>& deep_model<Loss>::operator+=(model_layer&& layer)
	{
		layer_dict_[layer->name] = depth_;
		layers_[depth_] = std::move(layer);
		++depth_;

		return *this;
	}

	template <typename Loss>
	deep_model<Loss>& deep_model<Loss>::operator-=(std::string layer_name)
	{
		if (layer_dict_.find(layer_name) == layer_dict_.end())
			throw std::runtime_error("Layer with given name doesn't exists in current model!");

		auto const index = layer_dict_[layer_name];
		for (auto i = index; i < depth_ - 1; ++i)
		{
			layers_[i] = std::move(layers_[i + 1]);
			layer_dict_[layers_[i]->name] = i;
		}

		layers_.erase(depth_ - 1);
		layer_dict_.erase(layer_name);
		--depth_;

		return *this;
	}

	template <typename Loss>
	model_layer& deep_model<Loss>::operator[](std::string name)
	{
		if (layer_dict_.find(name) == layer_dict_.end())
			throw std::runtime_error("Layer with given name doesn't exists!");
		return layers_[layer_dict_[name]];
	}

	template <typename Loss>
	model_layer& deep_model<Loss>::operator[](size_t index)
	{
		if (layers_.find(index) == layers_.end())
			throw std::runtime_error("Layer with given index doesn't exists!");
		return layers_[index];
	}

	template <typename Loss>
	tensor<float> deep_model<Loss>::forward_pass(tensor<float> const& x)
	{
		auto output = x;
		for (size_t i = 0; i < depth_; ++i)
			output = std::move(layers_[i]->forward_inputs(std::move(output)));

		return output;
	}

	template <typename Loss>
	tensor<float> deep_model<Loss>::feed_forward(tensor<float> const& x)
	{
		auto output = x;
		for (size_t i = 0; i < depth_; ++i)
			output = std::move(layers_[i]->forward(std::move(output)));

		return output;
	}

	template <typename Loss>
	std::vector<backward_grads<float, float, float, float>> deep_model<Loss>::backward_pass(tensor<float> const& x,
		tensor<float> const& y_oh)
	{
		std::vector<backward_grads<float, float, float, float>> grads;
		auto grads_out = loss_.backward_inputs(x, y_oh);
		if (loss_.has_params)
			grads.push_back(loss_.backward_params());
		for (auto i = depth_ - 1; i > 0; --i)
		{
			if (layers_[i]->has_params)
				grads.push_back(layers_[i]->backward_params(grads_out));
			grads_out = std::move(layers_[i]->backward_inputs(std::move(grads_out)));
		}

		if (layers_[0]->has_params)
			grads.push_back(layers_[0]->backward_params(grads_out));

		return grads;
	}

	template <typename Loss>
	float deep_model<Loss>::loss(tensor<float> const& logits, tensor<float> const& y_oh)
	{
		return loss_.forward_inputs(logits, y_oh);
	}

	template <typename Loss>
	double deep_model<Loss>::accuracy(tensor<float> const& logits,
		tensor<float> const& y_oh)
	{
		auto y = logits.argmax(1);
		auto y_ = y_oh.argmax(1);

		double const n = y.shape().front();

		return transprecision_floating_point::produce<size_t, size_t, size_t>(y, y_, [] __device__(size_t y_f, size_t y_f_) { return y_f == y_f_ ? size_t(1) : size_t(0); }).sum() / n;
	}

	template <typename Loss>
	double deep_model<Loss>::accuracy_from_data(tensor<float> const& inputs,
		tensor<float> const& y_oh)
	{
		auto y = this->evaluate(inputs);
		auto y_ = y_oh.argmax(1);

		double const n = y.shape().front();

		return transprecision_floating_point::produce<size_t, size_t, size_t>(y, y_, [] __device__(size_t y_f, size_t y_f_) { return y_f == y_f_ ? size_t(1) : size_t(0); }).sum() / n;
	}

	template <typename Loss>
	tensor<size_t> deep_model<Loss>::evaluate(tensor<float> const& x)
	{
		return forward_pass(x).argmax(1);
	}
}
