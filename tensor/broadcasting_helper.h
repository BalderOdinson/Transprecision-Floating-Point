#pragma once
#include "tensor.h"
#include <algorithm>

namespace transprecision_floating_point
{
	struct broadcasting_helper
	{
		broadcasting_helper(tensor_shape broadcast_shape, tensor_shape shape);

		bool can_broadcast() const;
		size_t tensor_size() const;
		uint_fast8_t axis() const;

	private:
		tensor_shape broadcast_shape_;
		tensor_shape shape_;
		bool can_broadcast_;
		size_t tensor_size_;
		uint_fast8_t axis_;
	};
}

