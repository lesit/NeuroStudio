#pragma once

#include "HiddenLayer.h"

#include "NeuroData/model/AbstractProducerModel.h"

namespace np
{
	namespace network
	{
		class OutputLayer : public HiddenLayer
		{
		public:
			OutputLayer(neuro_u32 uid);
			virtual ~OutputLayer();

			virtual _binding_model_type GetBindingModelType() const override { return _binding_model_type::network_output_layer; }

			_layer_type GetLayerType() const override  { return _layer_type::output; }

			void ChangedBindingDataShape() override;

			bool AvailableConnectHiddenLayer() const override  { return false; }
			bool AvailableConnectOutputLayer() const override { return false; }

			bool AvailableChangeType() const override  { return false; }

			void EntryValidation() override {}

			bool HasActivation() const override { return true; }
			bool AvailableChangeActivation() const override;
			virtual _activation_type GetActivation() const override;

			neuro_u32 AvailableInputCount() const override	{ return 1; }
			bool AvailableSetSideInput(const HiddenLayer* input) const override { return false; }

			neuro_u32 GetLayerDataInfoVector(_layer_data_info_vector& info_vector) const override { return 0; }
			
			bool ReadLabelForTarget() const;
			bool IsClassifyLossType() const;

		protected:
			void OnInsertedInput(AbstractLayer* layer) override;

			tensor::TensorShape MakeOutTensorShape() const override;
		};
	}
}
