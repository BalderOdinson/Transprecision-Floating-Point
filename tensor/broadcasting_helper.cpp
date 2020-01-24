#include "broadcasting_helper.h"

namespace transprecision_floating_point
{
	broadcasting_helper::broadcasting_helper(
		tensor_shape broadcast_shape,
		tensor_shape shape) : 
		broadcast_shape_(std::move(broadcast_shape)),
		shape_(std::move(shape))
	{
		auto const broadcast_dimension = tensor_shape_dimension(broadcast_shape_);
		auto const tensor_dimension = tensor_shape_dimension(shape_);
		if (broadcast_dimension > tensor_dimension)
		{
			can_broadcast_ = false;
			return;
		}

		tensor_size_ = 1;

		if (broadcast_shape_.back() == 1)
		{
			axis_ = 0;
			for (int64_t i = broadcast_dimension - 2; i > -1; --i)
			{
				if (broadcast_shape_[i] != shape_[i])
				{
					can_broadcast_ = false;
					return;
				}
				tensor_size_ *= shape_[i];
			}

			can_broadcast_ = true;
			return;
		}

		if (broadcast_shape_.front() == 1)
		{
			axis_ = 1;
			for (size_t i = 1; i < broadcast_dimension; ++i)
			{
				if (broadcast_shape_[i] != shape_[i])
				{
					can_broadcast_ = false;
					return;
				}
				tensor_size_ *= shape_[i];
			}

			can_broadcast_ = true;
			return;
		}

		can_broadcast_ = false;

	}

	bool broadcasting_helper::can_broadcast() const
	{
		return can_broadcast_;
	}

	size_t broadcasting_helper::tensor_size() const
	{
		return tensor_size_;
	}

	uint_fast8_t broadcasting_helper::axis() const
	{
		return axis_;
	}
	
}
