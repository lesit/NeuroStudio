#pragma once

#include "HiddenLayerEngine.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class RecurrentLayerEngine : public HiddenLayerEngine
			{
			public:
				static RecurrentLayerEngine* CreateInstance(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const HiddenLayerEngine* prev_conn);

				virtual bool OnInitialized() override;

				void SetContinuationIDC(_TYPED_TENSOR_DATA<bool, sizeof(bool)>* cont_flag)
				{
					m_cont_idc = cont_flag;
				}

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

			protected:
				RecurrentLayerEngine(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const RecurrentLayerEngine* prev_conn);
				virtual ~RecurrentLayerEngine();

				virtual bool MustHaveInput() const override{ return false; }
				virtual bool SupportBackwardInnerWeight() const override{ return true; }

				neuro_u32 m_time_length;

				neuro_u32 m_hidden_size;
				neuro_u32 m_total_gate_size;
				neuro_u32 m_batch_total_gate_per_time;
				neuro_u32 m_batch_hidden_per_time;

				const RecurrentLayerEngine* m_prev_rnn;

				_TYPED_TENSOR_DATA<bool, sizeof(bool)>* m_cont_idc;	// sequence continuation indicators
				bool m_bSequanceInEpoch;
			};
		}
	}
}
