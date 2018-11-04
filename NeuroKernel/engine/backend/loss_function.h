#pragma once
#include "common.h"
#include "core/math_device.h"

#include "../../network/NeuralNetworkTypes.h"
#include "activations.h"

#include "core/MathCoreApi.h"

namespace np 
{
	using namespace network;
	namespace engine
	{
		namespace loss
		{
			class LossFunction : protected core::MathCoreApi
			{
			public:
				static LossFunction* CreateInstance(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance, network::_loss_type type, bool read_label_for_target)
				{
					if (pdtype == core::math_device_type::cuda)
						return CreateInstanceCUDA(cuda_instance, type, read_label_for_target);
					else
						return CreateInstanceCPU(type, read_label_for_target);
				}

				static LossFunction* CreateInstanceCUDA(core::cuda::CudaInstance* cuda_instance, network::_loss_type type, bool read_label_for_target);
				static LossFunction* CreateInstanceCPU(network::_loss_type type, bool read_label_for_target);

				virtual ~LossFunction(){}

				/*	MSE,					regression 에서 사용
					CrossEntropy,			binary classification에서 사용
					CrossEntropyMulticlass	multi classification에서 사용

					따라서, 사용 유형에 따라 골라줘야 한다. 
					그리고, 출력을 softmax로 했다면 자동으로 CrossEntropyMulticlass가 선택되도록 해야함! 의무는 아니지만, 권장
					*/
				virtual network::_loss_type type() const = 0;

				// gradient for a minibatch
				virtual neuron_error CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target) = 0;
				virtual bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) = 0;

				static neuro_float normalize_factor(neuro_u32 size)
				{
					if (size == 0)
						return 1.f;

					return 1 / neuro_float(size);	// 1이 아니고 batch_size. 즉, deta scale은 1/batch_size
				}

			protected:
				LossFunction(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance, bool read_label_for_target)
					: MathCoreApi(pdtype, cuda_instance), m_read_label_for_target(read_label_for_target)
				{}

				const bool m_read_label_for_target;
			};
		}
	}
}
