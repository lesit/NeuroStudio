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
			class PoolingLayerEngine : public HiddenLayerEngine
			{
			public:
				static PoolingLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				PoolingLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~PoolingLayerEngine();

				const wchar_t* GetLayerName() const override{
					switch ((network::_pooling_type)m_entry.pooling.type)
					{
					case network::_pooling_type::ave_pooling:
						return L"Ave Pooling";
					}
					return L"Max Pooling";
				}

				virtual bool OnInitialized() override;
			};
		}
	}
}
