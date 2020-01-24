#pragma once
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"
#include "types.h"

namespace transprecision_floating_point
{
	namespace simple_ml_lib
	{
		template<typename Type>
		struct mnist_dataset
		{
			explicit mnist_dataset(std::string const& mnist_path);
			tensor<Type> train_labels;
			tensor<Type> train_images;
			tensor<Type> test_labels;
			tensor<Type> test_images;
		};

		template <typename Type>
		mnist_dataset<Type>::mnist_dataset(std::string const& mnist_path)
		{
			auto dataset = mnist::read_dataset();
			//normalize_dataset(dataset);

			auto const train_n = dataset.training_images.size();
			auto const test_n = dataset.test_images.size();
			auto const features = dataset.test_images[0].size();

			train_labels = tensor<Type>({ train_n, 10 });
			train_images = tensor<Type>({ train_n, features });
			test_labels = tensor<Type>({ test_n, 10 });
			test_images = tensor<Type>({ test_n, features });

			#pragma omp parallel for
			for (int64_t i = 0; i < train_n; ++i)
			{
				train_labels[{i, dataset.training_labels[i]}] = Type(1);
				if (i < test_n)
					test_labels[{i, dataset.test_labels[i]}] = Type(1);
				for (size_t j = 0; j < features; ++j)
				{
					train_images[{i, j}] = Type(dataset.training_images[i][j]);
					if (i < test_n)
						test_images[{i, j}] = Type(dataset.training_images[i][j]);
				}
			}

			train_images /= Type(255.0);
			test_images /= Type(255.0);
		}
	}
}
