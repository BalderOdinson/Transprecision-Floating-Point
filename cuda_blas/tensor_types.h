#pragma once
#include <utility>
#include <memory>
#include <functional>

using tensor_shape = std::pair<size_t, size_t>;
using tensor_index = tensor_shape;
template <typename Type, size_t Height, size_t Width>
using array2d = std::array<std::array<Type, Width>, Height>;
template <typename Type>
using tensor_data = std::unique_ptr<Type[], std::function<void(Type*)>>;