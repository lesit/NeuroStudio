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
		// ������ layer �� �ϴ� ���� ���⿡���� �� ���� �׷��� layer ���� �����ϵ��� �ؼ�
		// �ϳ��� layer�� �Ѱ� �̻��� layer�� ������, �� layer ���� ������ layer ���� �κ� ������ �� �� �ֵ��� �Ѵ�.
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
