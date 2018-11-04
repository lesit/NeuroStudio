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
		// 보통은 layer 로 하는 것을 여기에서는 더 작은 그룹인 layer 으로 연결하도록 해서
		// 하나의 layer에 한개 이상의 layer을 가지고, 두 layer 간의 연결은 layer 간의 부분 연결을 할 수 있도록 한다.
		namespace layers
		{
			class ConcatLayerEngine : public HiddenLayerEngine
			{
			public:
				static ConcatLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				ConcatLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~ConcatLayerEngine();

				const wchar_t* GetLayerName() const override{
					return L"Concat";
				}

				virtual bool OnInitialized() override;
				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

			protected:
				virtual bool HasOneInput() const { return false; }

				virtual bool Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output_buffer) override;
				virtual bool Backward(neuro_u32 batch_size) override;

				neuro_8 m_concat_axis;
				neuro_u32 m_num_concats;
				neuro_u32 m_concat_input_size;

				neuro_u32 m_output_concat_axis_size;
			};
		}
	}
}
