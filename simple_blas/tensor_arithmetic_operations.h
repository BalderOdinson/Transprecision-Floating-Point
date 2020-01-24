#pragma once

template <typename Type>
tensor<Type>& tensor<Type>::transpose()
{
	auto const copy = *this;
	
	for (int64_t n = 0; n < shape_.first * shape_.second; ++n)
	{
		auto i = n / shape_.first;
		auto j = n % shape_.first;
		data_[n] = copy.data_[j * shape_.second + i];
	}

	std::swap(shape_.first, shape_.second);

	return *this;
}

template <typename Type>
tensor<Type> transpose(tensor<Type> tensor)
{
	return tensor.transpose();
}

template <typename Type>
tensor<Type>& tensor<Type>::dot(tensor<Type> const& other)
{
	if (shape_.second != other.shape_.first)
		throw std::runtime_error("Invalid shapes - " + to_string(shape_) + " and " + to_string(other.shape_) + "!");

	auto const n = shape_.first;
	auto const m = shape_.second;
	auto const p = other.shape_.second;

	auto const other_transposed = std::move(simple_blas::transpose(other));

	tensor<Type> result;
	result.shape_ = { n, p };
	result.data_ = std::make_unique<Type[]>(n * p);

#pragma omp parallel for
	for (int64_t i = 0; i < n; ++i)
	{
		for (int64_t j = 0; j < p; ++j)
		{
			Type dot = 0;
			for (int64_t k = 0; k < m; ++k)
				dot += data_[i * m + k] * other_transposed.data_[j * m + k];

			result.data_[i * p + j] = dot;
		}
	}

	return (*this = std::move(result));
}

template <typename Type>
tensor<Type> dot(tensor<Type> lhs, tensor<Type> const& rhs)
{
	auto result = lhs.dot(rhs);
	return result;
}

template <typename Type>
template<uint_fast8_t Axis>
tensor<Type> tensor<Type>::sum() const
{
	if constexpr (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		for (int64_t i = 0; i < shape_.first; ++i)
			for (int64_t j = 0; j < shape_.second; ++j)
				result.data_[j] += data_[i * shape_.second + j];

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
			for (int64_t j = 0; j < shape_.second; ++j)
				result.data_[i] += data_[i * shape_.second + j];

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
Type tensor<Type>::sum() const
{
	Type sum = 0;
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			sum += data_[i * shape_.second + j];
	return sum;
}

template <typename Type>
template <uint_fast8_t Axis>
tensor<Type> tensor<Type>::max() const
{
	if constexpr (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		for (int64_t j = 0; j < shape_.second; ++j)
		{
			result.data_[j] = data_[j];
			for (int64_t i = 0; i < shape_.first; ++i)
				result.data_[j] = std::max(data_[i * shape_.second + j], result.data_[j]);
		}

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
		{
			result.data_[i] = data_[i];
			for (int64_t j = 0; j < shape_.second; ++j)
				result.data_[i] = std::max(data_[i * shape_.second + j], result.data_[i]);
		}

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
Type tensor<Type>::max() const
{
	Type max = data_[0];
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			max = std::max(data_[i * shape_.second + j], max);
	return max;
}

template <typename Type>
template <uint_fast8_t Axis>
tensor<Type> tensor<Type>::min() const
{
	if constexpr (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		for (int64_t j = 0; j < shape_.second; ++j)
		{
			result.data_[j] = data_[j];
			for (int64_t i = 0; i < shape_.first; ++i)
				result.data_[j] = std::min(data_[i * shape_.second + j], result.data_[j]);
		}

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
		{
			result.data_[i] = data_[i];
			for (int64_t j = 0; j < shape_.second; ++j)
				result.data_[i] = std::min(data_[i * shape_.second + j], result.data_[i]);
		}

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
Type tensor<Type>::min() const
{
	Type min = data_[0];
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			min = std::min(data_[i * shape_.second + j], min);
	return min;
}


template <typename Type>
template <uint_fast8_t Axis>
tensor<size_t> tensor<Type>::argmax() const
{
	if constexpr (Axis == 0)
	{
		tensor<size_t> result({ 1, shape_.second });

		for (int64_t j = 0; j < shape_.second; ++j)
		{
			result[{0, j}] = 0;
			for (int64_t i = 0; i < shape_.first; ++i)
			{
				auto const current = result[{0, j}];
				result[{0, j}] = data_[i * shape_.second + j] > data_[current * shape_.second + j] ? i : current;
			}
		}

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<size_t> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
		{
			result[{i, 0}] = 0;
			for (int64_t j = 0; j < shape_.second; ++j)
			{
				auto const current = result[{i, 0}];
				result[{i, 0}] = data_[i * shape_.second + j] > data_[i * shape_.second + current] ? j : current;
			}
		}

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
tensor_index tensor<Type>::argmax() const
{
	tensor_index max = { 0,0 };
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			if ((data_[i * shape_.second + j] > data_[max.first * shape_.second + max.second])) max = tensor_index{ i, j };
	return max;
}

template <typename Type>
template <uint_fast8_t Axis>
tensor<size_t> tensor<Type>::argmin() const
{
	if constexpr (Axis == 0)
	{
		tensor<size_t> result({ 1, shape_.second });

		for (int64_t j = 0; j < shape_.second; ++j)
		{
			result[{0, j}] = 0;
			for (int64_t i = 0; i < shape_.first; ++i)
			{
				auto const current = result[{0, j}];
				result[{0, j}] = data_[i * shape_.second + j] < data_[current * shape_.second + j] ? i : current;
			}
		}

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<size_t> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
		{
			result[{i, 0}] = data_[i];
			for (int64_t j = 0; j < shape_.second; ++j)
			{
				auto const current = result[{i, 0}];
				result[{i, 0}] = data_[i * shape_.second + j] < data_[i * shape_.second + current] ? j : current;
			}
		}

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
tensor_index tensor<Type>::argmin() const
{
	tensor_index min = { 0,0 };
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			if ((data_[i * shape_.second + j] < data_[min.first * shape_.second + min.second])) min = tensor_index{ i, j };
	return min;
}

template <typename Type>
template <typename Reduction, uint_fast8_t Axis>
tensor<Type> tensor<Type>::reduce(Reduction reduction) const
{
	if constexpr (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		for (int64_t j = 0; j < shape_.second; ++j)
		{
			result.data_[j] = data_[j];
			for (int64_t i = 1; i < shape_.first; ++i)
				result.data_[j] = reduction(data_[i * shape_.second + j], result.data_[j]);
		}

		return result;
	}

	if constexpr (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		for (int64_t i = 0; i < shape_.first; ++i)
		{
			result.data_[i] = data_[i * shape_.second];
			for (int64_t j = 1; j < shape_.second; ++j)
				result.data_[i] = reduction(data_[i * shape_.second + j], result.data_[i]);
		}

		return result;
	}

	throw std::runtime_error("Invalid axis template parameter!");
}

template <typename Type>
template <typename Reduction>
Type tensor<Type>::reduce(Reduction reduction) const
{
	Type r = data_[0];
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = i == 0 ? 1 : 0; j < shape_.second; ++j)
			r = reduction(data_[i * shape_.second + j], r);
	return r;
}

template <typename Type>
tensor<Type>& tensor<Type>::log()
{
#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::log(static_cast<double>(data_[i * shape_.second + j]));

	return *this;
}

template <typename Type>
tensor<Type> log(tensor<Type> tensor)
{
	return tensor.log();
}

template <typename Type>
tensor<Type>& tensor<Type>::exp()
{
	#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::exp(static_cast<double>(data_[i * shape_.second + j]));

	return *this;
}

template <typename Type>
tensor<Type> exp(tensor<Type> tensor)
{
	return tensor.exp();
}

template <typename Type>
tensor<Type>& tensor<Type>::abs()
{
	#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::fabs(static_cast<double>(data_[i * shape_.second + j]));

	return *this;
}

template <typename Type>
tensor<Type> abs(tensor<Type> tensor)
{
	return tensor.abs();
}


template <typename Type>
tensor<Type>& tensor<Type>::sqrt()
{
	#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::sqrt(static_cast<double>(data_[i * shape_.second + j]));

	return *this;
}

template <typename Type>
tensor<Type> sqrt(tensor<Type> tensor)
{
	return tensor.sqrt();
}

template <typename Type>
tensor<Type>& tensor<Type>::maximum(Type max)
{
	#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::max(data_[i * shape_.second + j], max);

	return *this;
}

template <typename Type>
tensor<Type> maximum(tensor<Type> tensor, Type max)
{
	return tensor.maximum(max);
}

template <typename Type>
tensor<Type>& tensor<Type>::minimum(Type max)
{
#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = std::min(data_[i * shape_.second + j], max);

	return *this;
}

template <typename Type>
tensor<Type> minimum(tensor<Type> tensor, Type max)
{
	return tensor.minimum(max);
}

template <typename Type>
template <typename Function>
tensor<Type> tensor<Type>::apply(Function func)
{
	#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (int64_t j = 0; j < shape_.second; ++j)
			data_[i * shape_.second + j] = func(data_[i * shape_.second + j]);

	return *this;
}

template <typename Type, typename Function>
tensor<Type> apply(tensor<Type> tensor, Function func)
{
	return tensor.apply(func);
}

template<typename Type, typename Function>
tensor<Type> produce(tensor<Type> lhs, tensor<Type> const& rhs, Function function)
{
	if (lhs.shape().first == rhs.shape().first && rhs.shape().second == 1)
	{
		#pragma omp parallel for
		for (int64_t i = 0; i < lhs.shape().first; ++i)
		{
			auto const elem = rhs[{i, 0}];
			for (int64_t j = 0; j < lhs.shape().second; ++j)
				lhs[{ i, j }] = Type(function(lhs[{ i, j }], elem));
		}
	}
	else if (lhs.shape().second == rhs.shape().second && rhs.shape().first == 1)
	{
		#pragma omp parallel for
		for (int64_t j = 0; j < lhs.shape().second; ++j)
		{
			auto const elem = rhs[{0, j}];
			for (int64_t i = 0; i < lhs.shape().first; ++i)
				lhs[{ i, j }] = Type(function(lhs[{ i, j }], elem));
		}
	}
	else if (lhs.shape().first == rhs.shape().first && lhs.shape().second == rhs.shape().second)
	{
		#pragma omp parallel for
		for (int64_t i = 0; i < lhs.shape().first; ++i)
			for (int64_t j = 0; j < lhs.shape().second; ++j)
				lhs[{ i, j }] = Type(function(lhs[{ i, j }], rhs[{i, j}]));
	}
	else
		throw std::runtime_error("Invalid shapes - " + to_string(lhs.shape()) + " and " + to_string(rhs.shape()) + "!");

	return lhs;
}