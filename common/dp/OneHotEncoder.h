#if !defined(_ONE_HOT_ENCODER_H)
#define _ONE_HOT_ENCODER_H

#include "common.h"

namespace np
{
	class OneHotEncoder
	{
	public:
		inline static void Encoding(neuro_u32 value, neuron_value* buffer, neuro_u32 size)
		{
			memset(buffer, 0, sizeof(neuron_value)*size);

			if (value<size)
				buffer[value] = 1;
		}
	};
}
#endif
