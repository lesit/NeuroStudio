#pragma once

#include "RecurrentLayerEngine.h"

/*
	This source came from junhyukoh/caffe-lstm(https://github.com/junhyukoh/caffe-lstm)
	and has been modified for Neuro Studio structure
*/
namespace np
{
	namespace engine
	{
		namespace layers
		{
			/*	Long short term memory (LSTM)
			*/
			class LstmLayerEngine : public RecurrentLayerEngine
			{
			public:
				static LstmLayerEngine* CreateInstance(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const RecurrentLayerEngine* prev_conn);

				LstmLayerEngine(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const RecurrentLayerEngine* prev_conn);
				virtual ~LstmLayerEngine();

				const wchar_t* GetLayerName() const override{
					return L"LSTM";
				}

				virtual neuro_u32 Get1MultiplierSizePerBatch() const override;

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

				const _NEURO_TENSOR_DATA& GetLastCellData() const{
					return m_last_time_cell;
				}
				const _NEURO_TENSOR_DATA& GetLastCellDiff() const{
					return m_last_time_cell_diff;
				}
				const _NEURO_TENSOR_DATA& GetLastHiddenData() const{
					return m_last_time_hidden;
				}
				const _NEURO_TENSOR_DATA& GetLastHiddenDiff() const{
					return m_last_time_hidden_diff;
				}
			protected:
				_ts_batch_time_order TensorBatchTimeOrder() const override{
					return _ts_batch_time_order::TxNxD;
				}

				_NEURO_TENSOR_DATA& GetCellDiff(neuro_u32 time)
				{
					return time % 2 == 0 ? m_cell_diff1 : m_cell_diff2;
				}

				neuro_float m_clipping_threshold; // threshold for clipped gradient

				_NEURO_TENSOR_DATA m_cell;				// memory cell
				_NEURO_TENSOR_DATA m_cell_diff1;		// memory cell diff
				_NEURO_TENSOR_DATA m_cell_diff2;

				_NEURO_TENSOR_DATA m_prev_gate_diff;		// gate values before nonlinearity
				_NEURO_TENSOR_DATA m_gate;			// gate values after nonlinearity

				_NEURO_TENSOR_DATA m_last_time_cell;		// next cell state value			c_T
				_NEURO_TENSOR_DATA m_last_time_cell_diff;	// next cell diff					c_T
				_NEURO_TENSOR_DATA m_last_time_hidden;		// next hidden activation value		h_T
				_NEURO_TENSOR_DATA m_last_time_hidden_diff;	// next hidden diff					h_T
			};
		}
	}
}
