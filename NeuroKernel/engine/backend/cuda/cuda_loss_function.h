#if !defined(_CUDA_LOSS_FUNCTION_H)
#define _CUDA_LOSS_FUNCTION_H

#include "../loss_function.h"

namespace np {
	namespace engine
	{
		namespace loss
		{
			namespace cuda
			{
				// mean-squared-error loss function for regression
				class CUDALossFunction : public LossFunction
				{
				public:
					CUDALossFunction(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target);
					virtual ~CUDALossFunction();

					virtual network::_loss_type type() const = 0;

					virtual neuron_error CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target) override;

				protected:
					virtual bool CalcLossVector(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuron_value* loss) {
						return false;
					}
				};

				class MSE : public CUDALossFunction
				{
				public:
					MSE(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target)
						: CUDALossFunction(cuda_instance, read_label_for_target){}

					virtual network::_loss_type type() const override{
						return network::_loss_type::MSE;
					}

					bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) override;
				protected:
					virtual bool CalcLossVector(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuron_value* loss) override;
				};

				// cross-entropy loss function for (multiple independent) binary classifications
				class CrossEntropy : public CUDALossFunction
				{
				public:
					CrossEntropy(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target);

					virtual network::_loss_type type() const override{
						return network::_loss_type::CrossEntropy;
					}

					bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) override;

				protected:
					virtual bool CalcLossVector(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuron_value* loss) override;
				};

				// SoftmaxWithLoss 를 하면 CalcDiff는 mse 처럼 output-target 만 해주고, softmax backward를 하지 않기 때문에 빠르다.\
				// 나중에 꼭 해봐야 함!
				// cross-entropy loss function for multi-class classification
				class CrossEntropyMulticlass : public CUDALossFunction
				{
				public:
					CrossEntropyMulticlass(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target);

					virtual network::_loss_type type() const override{
						return network::_loss_type::CrossEntropyMulticlass;
					}

					neuron_error CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target) override;
					bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) override;
				};
			}
		}
	}
}

#endif
