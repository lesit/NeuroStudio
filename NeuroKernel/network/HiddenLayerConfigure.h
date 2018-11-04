#pragma once

#include "NeuralNetworkTypes.h"

#include "HiddenLayer.h"
namespace np
{
	namespace network
	{
		/*	HiddenLayer�� ��ӹ��� �ʰ� �̷��� ���� �� ������
			��ӹ޾Ƽ� �� ConvLayer, FcLayer ���� �����ϸ� layer�� type�� �����Ҷ� ���ο� instance�� �����ϰ�
			������ ��/��µ� ���� �� network matrix�� �Ȱ��� ��ġ�� �ٽ� �־�� �Ѵ�.
			��, ����� ��ġ ������.. �Ѥ�
			����, ��/����� �����ϰ� ���븸 �����Ҽ� �ֵ��� �Ʒ��� ���� �Ͽ���.
		*/ 
		class HiddenLayerConfigure
		{
		public:
			virtual _layer_type GetLayerType() const = 0;
			
			virtual void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) {}

			virtual bool HasActivation() const { return false; }

			// activation�� ������ �������� ����
			virtual bool AvailableChangeActivation(const nsas::_LAYER_STRUCTURE_UNION& entry) const {
				return true;
			}

			// activation�� ������ �������� ����
			virtual _activation_type GetActivationType(const nsas::_LAYER_STRUCTURE_UNION& entry, _activation_type org_type) const {
				return org_type;
			}

			virtual neuro_u32 AvailableInputCount() const { return 1; }
			virtual bool AvailableSetSideInput(const HiddenLayer& layer, const HiddenLayer* input) const { return false; }
			virtual bool SetOutTensorShape(HiddenLayer& layer, const tensor::TensorShape& ts) { return false; }

			virtual tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const {
				return layer.GetMainInputTs();
			}

			virtual neuro_u32 GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const { return 0; }

			virtual bool HasWeight() const { return false; }

			virtual void SetLossType(nsas::_LAYER_STRUCTURE_UNION& entry, _loss_type type) {}
			virtual _loss_type GetLossType(const nsas::_LAYER_STRUCTURE_UNION& entry) const { return _loss_type::MSE; }
		};
	}
}
