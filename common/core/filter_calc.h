#if !defined(_FILTER_CALC_H)
#define _FILTER_CALC_H

#include "../common.h"

namespace np
{
	namespace core
	{
		inline static neuro_u32 filter_extent(neuro_u32 kernel, neuro_u32 dilation = 1)
		{
			return dilation * (kernel - 1) + 1;
		}

		inline static neuro_u32 filter_output_length(neuro_u32 input, neuro_u32 kernel, neuro_u32 stride, neuro_u32 dilation = 1, neuro_u32 pad_first = 0, neuro_u32 pad_last = 0)
		{
			const neuro_u32 fe = filter_extent(kernel, dilation);
			if (input + pad_first + pad_last < fe)
				return 0;

			return (input + pad_first + pad_last - fe) / stride + 1;
		}

		inline static std::pair<neuro_u32, neuro_u32> filter_pad(_pad_type pad_type, neuro_u32 input, neuro_u16 kernel, neuro_u16 stride, neuro_u32 output)
		{
			// sampe 모드에서 2로 나누어야 하므로 ceil로 한다. 이렇게 하면 stride가 어떻게 되든 
			// output = (neuro_u32)ceil(double(input) / double(stride)) = (input + 2 * pad - kernel) / stride + 1 이 된다.
			// (input + 2 * pad - kernel) / stride + 1 는 pad를 알고 있을때 output 구하는 계산식
			if (pad_type == _pad_type::same)
			{
				neuro_u32 pad = (output - 1) * stride + kernel - input;
				neuro_u32 left = pad / 2;

				return std::pair<neuro_u32, neuro_u32>(left, pad - left);
			}
			else
				return std::pair<neuro_u32, neuro_u32>(0, 0);
		}

		inline static neuro_u32 filter_output_length_mode(_pad_type pad_type, neuro_u32 input, neuro_u16 kernel, neuro_u16 stride, neuro_u32 dilation = 1)
		{
			std::pair<neuro_u32, neuro_u32> pad;
			if (pad_type == _pad_type::same)
			{
				neuro_u32 target_size = (neuro_u32)ceil(double(input) / double(stride));
				pad = filter_pad(pad_type, input, kernel, stride, target_size);
			}
			else if (pad_type == _pad_type::valid)
			{
				pad.first = pad.second = 0;
			}
			else
				return 0;
			return filter_output_length(input, kernel, stride, dilation, pad.first, pad.second);
		}
	}
}
#endif
