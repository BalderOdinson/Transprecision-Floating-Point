#pragma once

template <typename Type>
tensor<Type>& tensor<Type>::transpose()
{
	auto const copy = *this;

	cuda_blas::_transpose(copy.data_, data_, shape_);

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

	auto const other_transposed = std::move(cuda_blas::transpose(other));

	tensor<Type> result;
	result.shape_ = { n, p };
	result.data_ = create_tensor_data<Type>(n * p);

	cuda_blas::_matrix_multiply(data_, other_transposed.data_, result.data_, n, m, p);

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
	if  (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		auto s = [] __device__(Type first, Type second) { return first + second; };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

		return result;
	}

	if  (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		auto s = [] __device__(Type first, Type second) { return first + second; };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

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
	if  (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		auto s = [] __device__(Type first, Type second) { return fmaxf(float(first), float(second)); };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

		return result;
	}

	if  (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		auto s = [] __device__(Type first, Type second) { return fmaxf(float(first), float(second)); };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

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
	if  (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });

		auto s = [] __device__(Type first, Type second) { return fminf(float(first), float(second)); };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

		return result;
	}

	if  (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });

		auto s = [] __device__(Type first, Type second) { return fminf(float(first), float(second)); };
		cuda_blas::_aggregate<Type, Axis, decltype(s)>(data_, result.data_, shape_, s);

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
	if  (Axis == 0)
	{
		tensor<size_t> result({ 1, shape_.second });
		cuda_blas::_argmax<Type, Axis>(data_, result.data_, shape_);
		return result;
	}

	if  (Axis == 1)
	{
		tensor<size_t> result({ shape_.first, 1 });
		cuda_blas::_argmax<Type, Axis>(data_, result.data_, shape_);
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
	if  (Axis == 0)
	{
		tensor<size_t> result({ 1, shape_.second });
		cuda_blas::_argmin<Type, Axis>(data_, result.data_, shape_);
		return result;
	}

	if  (Axis == 1)
	{
		tensor<size_t> result({ shape_.first, 1 });
		cuda_blas::_argmin<Type, Axis>(data_, result.data_, shape_);
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
	if  (Axis == 0)
	{
		tensor<Type> result({ 1, shape_.second });
		cuda_blas::_aggregate<Type, Axis, Reduction>(data_, result.data_, shape_, reduction);
		return result;
	}

	if  (Axis == 1)
	{
		tensor<Type> result({ shape_.first, 1 });
		cuda_blas::_aggregate<Type, Axis, Reduction>(data_, result.data_, shape_, reduction);
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
	cuda_blas::_apply(data_, [] __device__ (Type value) { return logf(float(value)); }, shape_);
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
	cuda_blas::_apply(data_, [] __device__ (Type value) { return expf(float(value)); }, shape_);
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
	cuda_blas::_apply(data_, [] __device__(Type value) { return fabsf(float(value)); }, shape_);
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
	cuda_blas::_apply(data_, [] __device__(Type value) { return sqrtf(float(value)); }, shape_);
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
	cuda_blas::_apply(data_, [max] __device__(Type value) { return fmaxf(float(value), float(max)); }, shape_);
	return *this;
}

template <typename Type>
tensor<Type> maximum(tensor<Type> tensor, Type max)
{
	return tensor.maximum(max);
}

template <typename Type>
tensor<Type>& tensor<Type>::minimum(Type min)
{
	cuda_blas::_apply(data_, [min] __device__(Type value) { return fminf(float(value), float(min)); }, shape_);
	return *this;
}

template <typename Type>
tensor<Type> minimum(tensor<Type> tensor, Type min)
{
	return tensor.minimum(min);
}

template <typename Type>
template <typename Function>
tensor<Type> tensor<Type>::apply(Function func)
{
	cuda_blas::_apply(data_, func, shape_);
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
		cuda_blas::_produce<Type, Function, 0>(lhs.data_, rhs.data_, lhs.data_, function, lhs.shape_);
	else if (lhs.shape().second == rhs.shape().second && rhs.shape().first == 1)
		cuda_blas::_produce<Type, Function, 1>(lhs.data_, rhs.data_, lhs.data_, function, lhs.shape_);
	else if (lhs.shape().first == rhs.shape().first && lhs.shape().second == rhs.shape().second)
		cuda_blas::_produce(lhs.data_, rhs.data_, lhs.data_, function, lhs.shape_);
	else
		throw std::runtime_error("Invalid shapes - " + to_string(lhs.shape()) + " and " + to_string(rhs.shape()) + "!");

	return lhs;
}