#pragma once

#include "NeuralNetworkTypes.h"

#include "HiddenLayer.h"
namespace np
{
	namespace network
	{
		/*	HiddenLayer를 상속받지 않고 이렇게 따로 한 이유는
			상속받아서 각 ConvLayer, FcLayer 등을 구현하면 layer의 type을 변경할때 새로운 instance를 생성하고
			기존의 입/출력들 적용 및 network matrix의 똑같은 위치에 다시 넣어야 한다.
			즉, 상당히 골치 아프다.. ㅡㅡ
			따라서, 입/출력을 제외하고 내용만 변경할수 있도록 아래와 같이 하였다.
		*/ 
		class HiddenLayerConfigure
		{
		public:
			virtual _layer_type GetLayerType() const = 0;
			
			virtual void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) {}

			virtual bool HasActivation() const { return false; }

			// activation을 가지고 있을때의 기준
			virtual bool AvailableChangeActivation(const nsas::_LAYER_STRUCTURE_UNION& entry) const {
				return true;
			}

			// activation을 가지고 있을때의 기준
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
