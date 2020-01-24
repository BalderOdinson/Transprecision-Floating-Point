#pragma once

template<typename Type>
void delete_tensor_data(Type* data)
{
	transprecision_floating_point::cuda_blas::_process_cuda_error(cudaDeviceSynchronize());
	transprecision_floating_point::cuda_blas::_process_cuda_error(cudaFree(data));
}

template<typename Type>
tensor_data<Type> create_tensor_data(size_t size)
{
	Type* data;
	transprecision_floating_point::cuda_blas::_process_cuda_error(cudaMallocManaged(&data, size * sizeof(Type)));
	transprecision_floating_point::cuda_blas::_process_cuda_error(cudaDeviceSynchronize());
	return tensor_data<Type>(data, delete_tensor_data<Type>);
}

template <typename Type>
tensor<Type>::tensor(tensor<Type> const& other) :
	shape_(other.shape_),
	default_element_(other.default_element_)
{
	data_ = create_tensor_data<Type>(other.shape_.first * other.shape_.second);

	cuda_blas::_copy(other.data_, data_, shape_);
}

template <typename Type>
tensor<Type>::tensor(tensor<Type>&& other) noexcept :
	shape_(std::move(other.shape_)),
	data_(std::move(other.data_)),
	default_element_(other.default_element_)
{
	other.data_ = nullptr;
}

template <typename Type>
tensor<Type>::tensor(tensor_shape shape, Type default_element) : shape_(shape), data_(create_tensor_data<Type>(shape.first * shape.second)), default_element_(default_element)
{
	cuda_blas::_initialize(data_, default_element_, shape_);
}

template <typename Type>
tensor<Type>& tensor<Type>::operator=(tensor<Type> const& other)
{
	return (*this = tensor<Type>(other));
}

template <typename Type>
tensor<Type>& tensor<Type>::operator=(tensor<Type>&& other) noexcept
{
	std::swap(shape_, other.shape_);
	std::swap(data_, other.data_);
	std::swap(default_element_, other.default_element_);

	return *this;
}

template <typename Type>
template <typename T>
tensor<T> tensor<Type>::cast()
{
	tensor<T> result({ shape_.first, shape_.second });

	cuda_blas::_cast(data_, result.data_, shape_);

	return result;
}

template <typename Type>
Type& tensor<Type>::operator[](tensor_index idx)
{
	return data_[idx.first * shape_.second + idx.second];
}

template <typename Type>
Type const& tensor<Type>::operator[](tensor_index idx) const
{
	return data_[idx.first * shape_.second + idx.second];
}


template <typename Type>
std::vector<tensor<Type>> tensor<Type>::split(size_t count)
{
	auto batch_size = shape_.first / count;
	auto shift = batch_size;
	std::vector<tensor<Type>> result(count);
	for (auto i = 0; i < count; ++i)
	{
		if (i == count - 1)
			batch_size = shape_.first - i * batch_size;
		tensor<Type> batch;
		batch.shape_ = { batch_size, shape_.second };
		batch.data_ = create_tensor_data<Type>(batch.shape_.first * batch.shape_.second);
		cuda_blas::_copy(data_.get() + shape_.second * i * shift, batch.data_.get(), batch.shape_);
		result[i] = std::move(batch);
	}

	return result;
}

template <typename Type>
tensor_shape tensor<Type>::shape() const
{
	return shape_;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::create(std::initializer_list<Type> data)
{
	auto const data_size = data.size();

	if (data_size > 1 && data_size % TensorWidth != 0)
		throw std::runtime_error("Invalid data shape!");

	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = create_tensor_data<Type>(TensorHeight * TensorWidth);

	if (data_size == 0)
	{
		cuda_blas::_initialize(t.data_, t.default_element_, t.shape_);
		return t;
	}

	if (data_size == 1)
	{
		t.default_element_ = *data.begin();

		cuda_blas::_initialize(t.data_, t.default_element_, t.shape_);
		return t;
	}

	if (data_size < TensorHeight * TensorWidth)
	{
		auto const copy_line = data_size / TensorWidth;

		cuda_blas::_copy_from_list(data, t.data_, t.shape_, copy_line);
	}
	else
	{
		cuda_blas::_copy_from_list(data, t.data_, t.shape_, TensorHeight);
	}

	return t;
}

template <typename Type>
tensor<Type> tensor<Type>::create(size_t tensor_height, size_t tensor_width, std::initializer_list<Type> data)
{
	auto const data_size = data.size();

	if (data_size > 1 && data_size % tensor_width != 0)
		throw std::runtime_error("Invalid data shape!");

	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = create_tensor_data<Type>(tensor_height * tensor_width);

	if (data_size == 0)
	{
		cuda_blas::_initialize(t.data_, t.default_element_, t.shape_);
		return t;
	}

	if (data_size == 1)
	{
		t.default_element_ = *data.begin();

		cuda_blas::_initialize(t.data_, t.default_element_, t.shape_);
		return t;
	}

	if (data_size < tensor_height * tensor_width)
	{
		auto const copy_line = data_size / tensor_width;

		cuda_blas::_copy_from_list(data, t.data_, t.shape_, copy_line);
	}
	else
		cuda_blas::_copy_from_list(data, t.data_, t.shape_, tensor_height);
	

	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::create(array2d<Type, TensorHeight, TensorWidth> const& data)
{
	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = cudaFree(data)(TensorHeight * TensorWidth);

	for (int64_t i = 0; i < TensorHeight; ++i)
		for (size_t j = 0; j < TensorWidth; ++j)
			t.data_[i * TensorWidth + j] = data[i][j];

	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth, typename Distribution>
tensor<Type> tensor<Type>::create_random(Distribution distribution)
{
	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = create_tensor_data<Type>(TensorHeight * TensorWidth);
	cuda_blas::_random_initialize(t.data_, distribution, t.shape_);
	return t;
}

template <typename Type>
template <typename Distribution>
tensor<Type> tensor<Type>::create_random(size_t tensor_height, size_t tensor_width, Distribution distribution)
{
	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = create_tensor_data<Type>(tensor_height * tensor_width);
	cuda_blas::_random_initialize(t.data_, distribution, t.shape_);
	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::identity()
{
	if (TensorHeight != TensorWidth)
		throw std::runtime_error("Invalid data shape!");

	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = create_tensor_data<Type>(TensorHeight * TensorWidth);


	cuda_blas::_initialize_diagonal({}, t.data_, t.shape_);

	return t;
}

template <typename Type>
tensor<Type> tensor<Type>::identity(size_t tensor_height, size_t tensor_width)
{
	if (tensor_height != tensor_width)
		throw std::runtime_error("Invalid data shape!");

	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = create_tensor_data<Type>(tensor_height * tensor_width);

	cuda_blas::_initialize_diagonal({}, t.data_, t.shape_);

	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::diagonal(std::array<Type, TensorHeight> const& data)
{
	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = create_tensor_data<Type>(TensorHeight * TensorWidth);

	cuda_blas::_initialize_diagonal(data, t.data_, t.shape_);

	return t;
}

template <typename Type>
tensor<Type> tensor<Type>::diagonal(size_t tensor_height, size_t tensor_width, std::initializer_list<Type> data)
{
	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = create_tensor_data<Type>(tensor_height * tensor_width);

	cuda_blas::_initialize_diagonal(data, t.data_, t.shape_);

	return t;
}