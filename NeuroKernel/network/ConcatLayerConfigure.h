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
				// concat_axis = -1 : time ���� ��� concat �Ѵ�.
				// concat_axis >= 0 : channel ���� concat�Ѵٴ� ��.
				// ����. concat_ts�� size �� 0�� ��� concat axis ���� ������ flatten �Ѵ�.
				/*	��, �Է��� [t=3, {2, 4, 3}] [t=2, {2, 4, 3}] [t=5, {2, 4, 3}] �� ���
					concat_axis=-1 �� �ǰ�, concat_ts={2, 4, 3} �� �ȴ�.
					����, ����� [t=10, {2, 4, 3}] �� �ȴ�.

					����,  �Է��� [t=3, {2, 7, 3}] [t=2, {5, 4, 3}] [t=5, {2, 4}] �� ���
					concat_axis=-1 �� �ǰ� concat_ts={} �� �ǹǷ�
					����, ����� [t = 3x2x7x3 + 2x5x4x3 + 5x2x4] �� �ȴ�.
					time�� ������ �ǹ̰� �������Ƿ�, [t=1, {3x2x7x3 + 2x5x4x3 + 5x2x4}] �� �ȴ�.
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
