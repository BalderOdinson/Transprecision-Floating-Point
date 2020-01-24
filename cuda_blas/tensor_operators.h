#pragma once

template <typename Type>
tensor<Type> tensor<Type>::operator+=(tensor<Type> const& other)
{
	if (shape().first == other.shape().first && other.shape().second == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first + second; };
		cuda_blas::_produce<Type, decltype(fun), 0>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().second == other.shape().second && other.shape().first == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first + second; };
		cuda_blas::_produce<Type, decltype(fun), 1>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().first == other.shape().first && shape().second == other.shape().second)
	{
		auto fun = [] __device__(Type first, Type second) { return first + second; };
		cuda_blas::_produce<Type, decltype(fun)>(data_, other.data_, data_, fun, shape_);
	}
	else
		throw std::runtime_error("Invalid shapes - " + to_string(shape()) + " and " + to_string(other.shape()) + "!");

	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator+=(Type other)
{
	cuda_blas::_apply(data_, [other] __device__(Type value) { return value + other; }, shape_);
	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator-=(tensor<Type> const& other)
{
	if (shape().first == other.shape().first && other.shape().second == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first - second; };
		cuda_blas::_produce<Type, decltype(fun), 0>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().second == other.shape().second && other.shape().first == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first - second; };
		cuda_blas::_produce<Type, decltype(fun), 1>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().first == other.shape().first && shape().second == other.shape().second)
	{
		auto fun = [] __device__(Type first, Type second) { return first - second; };
		cuda_blas::_produce<Type, decltype(fun)>(data_, other.data_, data_, fun, shape_);
	}
	else
		throw std::runtime_error("Invalid shapes - " + to_string(shape()) + " and " + to_string(other.shape()) + "!");

	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator-=(Type other)
{
	cuda_blas::_apply(data_, [other] __device__(Type value) { return value - other; }, shape_);
	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator*=(tensor<Type> const& other)
{
	if (shape().first == other.shape().first && other.shape().second == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first * second; };
		cuda_blas::_produce<Type, decltype(fun), 0>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().second == other.shape().second && other.shape().first == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first * second; };
		cuda_blas::_produce<Type, decltype(fun), 1>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().first == other.shape().first && shape().second == other.shape().second)
	{
		auto fun = [] __device__(Type first, Type second) { return first * second; };
		cuda_blas::_produce<Type, decltype(fun)>(data_, other.data_, data_, fun, shape_);
	}
	else
		throw std::runtime_error("Invalid shapes - " + to_string(shape()) + " and " + to_string(other.shape()) + "!");

	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator*=(Type other)
{
	cuda_blas::_apply(data_, [other] __device__(Type value) { return value * other; }, shape_);
	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator/=(tensor<Type> const& other)
{
	if (shape().first == other.shape().first && other.shape().second == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first / second; };
		cuda_blas::_produce<Type, decltype(fun), 0>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().second == other.shape().second && other.shape().first == 1)
	{
		auto fun = [] __device__(Type first, Type second) { return first / second; };
		cuda_blas::_produce<Type, decltype(fun), 1>(data_, other.data_, data_, fun, shape_);
	}
	else if (shape().first == other.shape().first && shape().second == other.shape().second)
	{
		auto fun = [] __device__(Type first, Type second) { return first / second; };
		cuda_blas::_produce<Type, decltype(fun)>(data_, other.data_, data_, fun, shape_);
	}
	else
		throw std::runtime_error("Invalid shapes - " + to_string(shape()) + " and " + to_string(other.shape()) + "!");

	return *this;
}

template <typename Type>
tensor<Type> tensor<Type>::operator/=(Type other)
{
	cuda_blas::_apply(data_, [other] __device__(Type value) { return value / other; }, shape_);
	return *this;
}

template<typename Type>
std::ostream& operator<<(std::ostream& os, tensor<Type> const& obj)
{
	os << "[";
	size_t i = 0;
	for (; i < obj.shape().first - 1; ++i)
	{
		os << "[";
		size_t j = 0;
		for (; j < obj.shape().second - 1; ++j)
			os << obj[{i, j}] << ", ";
		os << obj[{i, j}] << "]\n ";
	}
	os << "[";
	size_t j = 0;
	for (; j < obj.shape().second - 1; ++j)
		os << obj[{i, j}] << ", ";
	os << obj[{i, j}] << "]";

	os << "]";
	return os;
}


inline std::string to_string(tensor_shape shape)
{
	return "(" + std::to_string(shape.first) + "," + std::to_string(shape.second) + ")";
}

template<typename Type>
tensor<Type> operator+(tensor<Type> lhs, tensor<Type> const& rhs)
{
	lhs += rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator+(tensor<Type> lhs, Type rhs)
{
	lhs += rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator+(Type lhs, tensor<Type> rhs)
{
	rhs += lhs;
	return rhs;
}

template<typename Type>
tensor<Type> operator-(tensor<Type> lhs, tensor<Type> const& rhs)
{
	lhs -= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator-(tensor<Type> lhs, Type rhs)
{
	lhs -= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator-(Type lhs, tensor<Type> rhs)
{
	rhs -= lhs;
	return rhs;
}

template<typename Type>
tensor<Type> operator*(tensor<Type> lhs, tensor<Type> const& rhs)
{
	lhs *= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator*(tensor<Type> lhs, Type rhs)
{
	lhs *= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator*(Type lhs, tensor<Type> rhs)
{
	rhs *= lhs;
	return rhs;
}

template<typename Type>
tensor<Type> operator/(tensor<Type> lhs, tensor<Type> const& rhs)
{
	lhs /= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator/(tensor<Type> lhs, Type rhs)
{
	lhs /= rhs;
	return lhs;
}

template<typename Type>
tensor<Type> operator/(Type lhs, tensor<Type> rhs)
{
	rhs /= lhs;
	return rhs;
}