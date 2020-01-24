#pragma once
#include "cuda_error_defines.h"
#include "tensor_types.h"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_2D 32

namespace transprecision_floating_point
{
	struct cuda_tensor_dim
	{
		dim3 blocks_per_grid;
		dim3 threads_per_block;
		size_t n{};
		size_t m{};
	};


	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	inline cuda_tensor_dim _calculate_block_size(tensor_shape const& shape);
	inline cuda_tensor_dim _calculate_block_size_inverse(tensor_shape const& shape);

	/* ------------------------------------------------------------------------------------------
	| DECLARATIONS
	------------------------------------------------------------------------------------------ */

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */

	inline cuda_tensor_dim _calculate_block_size(tensor_shape const& shape)
	{
		auto const n = shape.front();
		auto const m = tensor_shape_total_size(shape) / n;

		dim3 threads_per_block(n, m);
		dim3 blocks_per_grid(1, 1);
		if (n*m > BLOCK_SIZE_2D)
		{
			threads_per_block.x = BLOCK_SIZE_2D;
			threads_per_block.y = BLOCK_SIZE_2D;
			blocks_per_grid.x = ceil(double(n) / double(threads_per_block.x));
			blocks_per_grid.y = ceil(double(m) / double(threads_per_block.y));
		}

		return { blocks_per_grid,threads_per_block, n, m };
	}

	inline cuda_tensor_dim _calculate_block_size_inverse(tensor_shape const& shape)
	{
		auto const m = shape.front();
		auto const n = tensor_shape_total_size(shape) / m;

		dim3 threads_per_block(n, m);
		dim3 blocks_per_grid(1, 1);
		if (n*m > BLOCK_SIZE_2D)
		{
			threads_per_block.x = BLOCK_SIZE_2D;
			threads_per_block.y = BLOCK_SIZE_2D;
			blocks_per_grid.x = ceil(double(n) / double(threads_per_block.x));
			blocks_per_grid.y = ceil(double(m) / double(threads_per_block.y));
		}

		return { blocks_per_grid,threads_per_block, n, m };
	}

	/* ------------------------------------------------------------------------------------------
	| DEFINITIONS
	------------------------------------------------------------------------------------------ */
}