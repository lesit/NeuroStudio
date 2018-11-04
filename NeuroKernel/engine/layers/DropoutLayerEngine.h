#pragma once

#include "HiddenLayerEngine.h"

/*
This source came from Caffe(https://github.com/BVLC/caffe)
and has been modified for Neuro Studio structure
*/
namespace np
{
	namespace engine
	{
		namespace layers
		{
			class DropoutLayerEngine : public HiddenLayerEngine
			{
			public:
				static DropoutLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				DropoutLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~DropoutLayerEngine();

				const wchar_t* GetLayerName() const override{
					return L"Dropout";
				}

				virtual bool OnInitialized() override;
				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

			protected:
				neuro_float m_dropout_scale;
				neuro_u32 m_uint_threshold;

				_TYPED_TENSOR_DATA<neuro_u32, sizeof(neuro_u32)> m_mask;
			};
		}
	}
}
