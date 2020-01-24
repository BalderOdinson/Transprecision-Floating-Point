#pragma once
#include "layer.h"

namespace transprecision_floating_point
{
	struct shared_workspace_memory
	{
		static size_t workspace_size;
		static workspace_memory memory;
		static void release();
	};

	size_t shared_workspace_memory::workspace_size = 0;
	workspace_memory shared_workspace_memory::memory = nullptr;

	inline void shared_workspace_memory::release()
	{
		if(memory)
		{
			memory.release();
			workspace_size = 0;
		}
	}

}

