#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class FcLayerConfigure : public HiddenLayerConfigure
		{
		public:
			network::_layer_type GetLayerType() const override { return network::_layer_type::fully_connected; }

			bool HasActivation() const override { return true; }
			bool HasWeight() const override { return true; }

			tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const override
			{
				return tensor::TensorShape(1, layer.GetEntry().fc.output_count, 1, 1);
			}

			bool SetOutTensorShape(HiddenLayer& layer, const tensor::TensorShape& ts) override
			{
				nsas::_LAYER_STRUCTURE_UNION entry = layer.GetEntry();
				entry.fc.output_count = ts.GetDimSize();
				layer.ChangeEntry(entry);
				return true;
			}

			neuro_u32 GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const override
			{
				const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

				info_vector.resize(2);
				info_vector[0] = { _layer_data_type::weight, entry.fc.output_count * layer.GetMainInputTs().GetTensorSize() };
				info_vector[1] = { _layer_data_type::bias, entry.fc.output_count };
				return info_vector.size();
			}
		};
	}
}
