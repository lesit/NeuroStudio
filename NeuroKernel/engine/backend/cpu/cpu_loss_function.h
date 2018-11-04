#if !defined(_CPU_LOSS_FUNCTION_H)
#define _CPU_LOSS_FUNCTION_H

#include "../loss_function.h"

namespace np {
	namespace engine
	{
		namespace loss
		{
			namespace cpu
			{
				// mean-squared-error loss function for regression
				class CPULossFunction : public LossFunction
				{
				public:
					CPULossFunction(bool read_label_for_target) : LossFunction(core::math_device_type::cpu, NULL, read_label_for_target){}
					virtual ~CPULossFunction(){}

					virtual network::_loss_type type() const = 0;
				};

				// Euclidean
				class MSE : public CPULossFunction
				{
				public:
					MSE(bool read_label_for_target)
					: CPULossFunction(read_label_for_target){}

					virtual network::_loss_type type() const override{
						return network::_loss_type::MSE;
					}

					neuron_error CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target) override;
					bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) override;
				};

				// cross-entropy loss function for (multiple independent) binary classifications
				class CrossEntropy : public CPULossFunction
				{
				public:
					CrossEntropy(bool read_label_for_target)
						: CPULossFunction(read_label_for_target) {}

					virtual network::_loss_type type() const override{
						return network::_loss_type::CrossEntropy;
					}

					neuron_error CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target) override;
					bool CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff) override;
				};

				// cross-entropy loss function for multi-class classification
				class CrossEntropyMulticlass : public CPULossFunction
				{
				public:
					CrossEntropyMulticlass(bool read_label_for_target)
						: CPULossFunction(read_label_for_target) {}

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
