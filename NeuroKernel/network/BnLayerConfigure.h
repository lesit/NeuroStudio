#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class BnLayerConfigure : public HiddenLayerConfigure
		{
		public:
			_layer_type GetLayerType() const override { return _layer_type::batch_norm; }

			void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) override
			{
				if (entry.batch_norm.momentum <= neuro_float(0) || entry.batch_norm.momentum >= neuro_float(1))
					entry.batch_norm.momentum = neuro_float(0.999);
			}

			bool HasActivation() const override { return true; }

			neuro_u32 GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const override
			{
				tensor::TensorShape in_ts = layer.GetMainInputTs();

				neuro_u32 channel = 1;
				if (in_ts.GetHeight() > 1 && in_ts.GetWidth() > 1)	// 3차원 이상으로 실제 channel이 있을 경우
					channel = in_ts.GetChannelCount();

				info_vector.resize(3);
				info_vector[0] = { _layer_data_type::other, channel };// save mean
				info_vector[1] = { _layer_data_type::other, channel };// save variance
				info_vector[2] = { _layer_data_type::other, 1 };// moving
				return info_vector.size();
			}
		};
	}
}
