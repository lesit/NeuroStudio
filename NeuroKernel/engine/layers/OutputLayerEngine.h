#pragma once

#include "HiddenLayerEngine.h"
#include "../backend/loss_function.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class OutputLayerEngine : public HiddenLayerEngine
			{
			public:
				static OutputLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				OutputLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~OutputLayerEngine();

				const wchar_t* GetLayerName() const { return L"Output Layer"; }

				bool Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output) override;
				bool Backward(neuro_u32 batch_size) override;

				_TYPED_TENSOR_DATA<void*, 4> GetTargetData() const { return m_target_buffer; }
				neuro_float GetLoss() const { return m_loss; }

			protected:
				virtual bool OnInitialized();
				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf);

			private:
				const bool m_read_label_for_target;

				loss::LossFunction* m_loss_function;
				loss::LossFunction* m_diff_function;

				bool m_is_backward_activation;

				_TYPED_TENSOR_DATA<void*, 4> m_target_buffer;
				neuro_float m_loss;
			};

			typedef std::vector<OutputLayerEngine*> _output_engine_vector;
		}
	}
}

