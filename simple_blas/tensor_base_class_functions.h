#pragma once

template <typename Type>
tensor<Type>::tensor(tensor<Type> const& other) :
	shape_(other.shape_),
	default_element_(other.default_element_)
{
	data_ = std::make_unique<Type[]>(other.shape_.first * other.shape_.second);

#pragma omp parallel for
	for (int64_t i = 0; i < other.shape_.first; ++i)
		for (size_t j = 0; j < other.shape_.second; ++j)
			data_[i * shape_.second + j] = other.data_[i * shape_.second + j];
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
tensor<Type>::tensor(tensor_shape shape, Type default_element) : shape_(shape), data_(new Type[shape.first * shape.second]), default_element_(default_element)
{
#pragma omp parallel for
	for (int64_t i = 0; i < shape.first; ++i)
		for (int64_t j = 0; j < shape.second; ++j)
			data_[i * shape.second + j] = default_element_;
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

#pragma omp parallel for
	for (int64_t i = 0; i < shape_.first; ++i)
		for (size_t j = 0; j < shape_.second; ++j)
			result[{i, j}] = static_cast<T>(data_[i * shape_.second + j]);

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
	t.data_ = std::make_unique<Type[]>(TensorHeight * TensorWidth);

	if (data_size == 0)
	{
		#pragma omp parallel for
		for (int64_t i = 0; i < TensorHeight; ++i)
			for (int64_t j = 0; j < TensorWidth; ++j)
				t.data_[i * TensorWidth + j] = t.default_element_;
		return t;
	}

	if (data_size == 1)
	{
		t.default_element_ = *data.begin();
#pragma omp parallel for
		for (int64_t i = 0; i < TensorHeight; ++i)
			for (int64_t j = 0; j < TensorWidth; ++j)
				t.data_[i * TensorWidth + j] = t.default_element_;
		return t;
	}

	if (data_size < TensorHeight * TensorWidth)
	{
		auto const copy_line = data_size / TensorWidth;
#pragma omp parallel for 
		for (int64_t i = 0; i < TensorHeight; ++i)
			for (size_t j = 0; j < TensorWidth; ++j)
				t.data_[i * TensorWidth + j] = *(data.begin() + (i * TensorWidth) % copy_line + j);
	}
	else
	{
#pragma omp parallel for
		for (int64_t i = 0; i < TensorHeight; ++i)
			for (size_t j = 0; j < TensorWidth; ++j)
				t.data_[i * TensorWidth + j] = *(data.begin() + (i * TensorWidth) + j);
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
	t.data_ = std::make_unique<Type[]>(tensor_height * tensor_width);

	if (data_size == 0)
	{
#pragma omp parallel for
		for (int64_t i = 0; i < tensor_height; ++i)
			for (int64_t j = 0; j < tensor_width; ++j)
				t.data_[i * tensor_width + j] = t.default_element_;
		return t;
	}

	if (data_size == 1)
	{
		t.default_element_ = *data.begin();
#pragma omp parallel for
		for (int64_t i = 0; i < tensor_height; ++i)
			for (int64_t j = 0; j < tensor_width; ++j)
				t.data_[i * tensor_width + j] = t.default_element_;
		return t;
	}

	if (data_size < tensor_height * tensor_width)
	{
		auto const copy_line = data_size / tensor_width;
#pragma omp parallel for 
		for (int64_t i = 0; i < tensor_height; ++i)
			for (size_t j = 0; j < tensor_width; ++j)
				t.data_[i * tensor_width + j] = *(data.begin() + (i * tensor_width) % copy_line + j);
	}
	else
	{
#pragma omp parallel for
		for (int64_t i = 0; i < tensor_height; ++i)
			for (size_t j = 0; j < tensor_width; ++j)
				t.data_[i * tensor_width + j] = *(data.begin() + (i * tensor_width) + j);
	}

	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::create(array2d<Type, TensorHeight, TensorWidth> const& data)
{
	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = new Type[TensorHeight * TensorWidth];

#pragma omp parallel for
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
	t.data_ = std::make_unique<Type[]>(TensorHeight * TensorWidth);

#pragma omp parallel for
	for (int64_t i = 0; i < TensorHeight; ++i)
		for (size_t j = 0; j < TensorWidth; ++j)
			t.data_[i * TensorWidth + j] = thread_safe_rand(distribution);

	return t;
}

template <typename Type>
template <typename Distribution>
tensor<Type> tensor<Type>::create_random(size_t tensor_height, size_t tensor_width, Distribution distribution)
{
	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = std::make_unique<Type[]>(tensor_height * tensor_width);

#pragma omp parallel for
	for (int64_t i = 0; i < tensor_height; ++i)
		for (size_t j = 0; j < tensor_width; ++j)
			t.data_[i * tensor_width + j] = thread_safe_rand(distribution);

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
	t.data_ = new Type[TensorHeight * TensorWidth];

#pragma omp parallel for
	for (int64_t i = 0; i < TensorHeight; ++i)
		for (size_t j = 0; j < TensorWidth; ++j)
			t.data_[i * TensorWidth + j] = i == j ? 1 : 0;

	return t;
}

template <typename Type>
tensor<Type> tensor<Type>::identity(size_t tensor_height, size_t tensor_width)
{
	if (tensor_height != tensor_width)
		throw std::runtime_error("Invalid data shape!");

	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = new Type[tensor_height * tensor_width];

#pragma omp parallel for
	for (int64_t i = 0; i < tensor_height; ++i)
		for (size_t j = 0; j < tensor_width; ++j)
			t.data_[i * tensor_width + j] = i == j ? 1 : 0;

	return t;
}

template <typename Type>
template <size_t TensorHeight, size_t TensorWidth>
tensor<Type> tensor<Type>::diagonal(std::array<Type, TensorHeight> const& data)
{
	tensor t;
	t.shape_ = { TensorHeight, TensorWidth };
	t.data_ = new Type[TensorHeight * TensorWidth];

#pragma omp parallel for
	for (int64_t i = 0; i < TensorHeight; ++i)
		for (size_t j = 0; j < TensorWidth; ++j)
			t.data_[i * TensorWidth + j] = i == j ? data[i] : 0;

	return t;
}

template <typename Type>
tensor<Type> tensor<Type>::diagonal(size_t tensor_height, size_t tensor_width, std::initializer_list<Type> data)
{
	tensor t;
	t.shape_ = { tensor_height, tensor_width };
	t.data_ = new Type[tensor_height * tensor_width];

#pragma omp parallel for
	for (int64_t i = 0; i < tensor_height; ++i)
		for (size_t j = 0; j < tensor_width; ++j)
			t.data_[i * tensor_width + j] = i == j ? data[i] : 0;

	return t;
}