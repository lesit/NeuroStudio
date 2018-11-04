#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		struct _CONCAT_INFO
		{
			_CONCAT_INFO()
			{
				// concat_axis = -1 : time 부터 모두 concat 한다.
				// concat_axis >= 0 : channel 부터 concat한다는 것.
				// 만약. concat_ts의 size 가 0인 경우 concat axis 이하 모든것을 flatten 한다.
				/*	즉, 입력이 [t=3, {2, 4, 3}] [t=2, {2, 4, 3}] [t=5, {2, 4, 3}] 인 경우
					concat_axis=-1 이 되고, concat_ts={2, 4, 3} 이 된다.
					따라서, 출력은 [t=10, {2, 4, 3}] 이 된다.

					만약,  입력이 [t=3, {2, 7, 3}] [t=2, {5, 4, 3}] [t=5, {2, 4}] 인 경우
					concat_axis=-1 이 되고 concat_ts={} 이 되므로
					따라서, 출력은 [t = 3x2x7x3 + 2x5x4x3 + 5x2x4] 가 된다.
					time만 있으면 의미가 없어지므로, [t=1, {3x2x7x3 + 2x5x4x3 + 5x2x4}] 이 된다.
				*/
				concat_axis = -1;
				concat_axis_size = 0;
			}
			_CONCAT_INFO(const _CONCAT_INFO& src)
			{
				*this = src;
			}
			_CONCAT_INFO& operator = (const _CONCAT_INFO& src)
			{
				join_ts = src.join_ts;
				concat_axis = src.concat_axis;
				concat_axis_size = src.concat_axis_size;
				concat_ts = src.concat_ts;
				return *this;
			}

			tensor::TensorShape toTensor() const
			{
				tensor::TensorShape ret = join_ts;
				if (concat_axis < 0)
					ret.time_length = concat_axis_size;
				else
					ret.push_back(concat_axis_size);
				ret.insert(ret.end(), concat_ts.begin(), concat_ts.end());
				return ret;
			}

			tensor::TensorShape join_ts;
			tensor::DataShape concat_ts;

			neuro_32 concat_axis;
			neuro_u32 concat_axis_size;
		};

		class ConcatLayerConfigure : public HiddenLayerConfigure
		{
		public:
			network::_layer_type GetLayerType() const override { return network::_layer_type::concat; }
			neuro_u32 AvailableInputCount() const override { return neuro_last32; }

			static _CONCAT_INFO GetConcatInfo(const HiddenLayer& layer);

			tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const override;
		};
	}
}
