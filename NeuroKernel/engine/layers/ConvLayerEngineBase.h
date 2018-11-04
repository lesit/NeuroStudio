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
			class ConvLayerEngineBase : public HiddenLayerEngine
			{
			public:
				static ConvLayerEngineBase* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				ConvLayerEngineBase(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~ConvLayerEngineBase();

				const wchar_t* GetLayerName() const override{
					return L"Convolutional";
				}

			protected:
				virtual bool OnInitialized() override;

				virtual inline bool reverse_dimensions() { return false; }	// if deconv, true

				std::pair<neuro_u32, neuro_u32> m_pad_height, m_pad_width;
			};
		}
	}
}
