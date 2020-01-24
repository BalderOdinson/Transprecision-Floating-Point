#pragma once
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "../tensor/tensor.h"

namespace transprecision_floating_point
{
	template<typename T>
	struct mnist_dataset
	{
		explicit mnist_dataset(std::string const& mnist_path);
		tensor<T> train_labels;
		tensor<T> train_images;
		tensor<T> test_labels;
		tensor<T> test_images;
	};

	template <typename T>
	mnist_dataset<T>::mnist_dataset(std::string const& mnist_path)
	{
		auto dataset = mnist::read_dataset();
		//normalize_dataset(dataset);

		auto const train_n = dataset.training_images.size();
		auto const test_n = dataset.test_images.size();
		auto const features = dataset.test_images[0].size();

		train_labels = tensor<T>({ train_n, 10 });
		train_images = tensor<T>({ train_n, features });
		test_labels = tensor<T>({ test_n, 10 });
		test_images = tensor<T>({ test_n, features });

		for (size_t i = 0; i < train_n; ++i)
		{
			train_labels[tensor_index({ i, dataset.training_labels[i] })] = T(1);
			if (i < test_n)
				test_labels[tensor_index({ i, dataset.test_labels[i] })] = T(1);
			for (size_t j = 0; j < features; ++j)
			{
				train_images[tensor_index({ i, j })] = T(dataset.training_images[i][j]);
				if (i < test_n)
					test_images[tensor_index({ i, j })] = T(dataset.test_images[i][j]);
			}
		}

		train_images /= T(255.0);
		test_images /= T(255.0);
	}
}
