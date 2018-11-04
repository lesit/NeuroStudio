#ifndef _CPU_ACTIVATION_FUNCTION_H
#define _CPU_ACTIVATION_FUNCTION_H

#include "../activations.h"

namespace np
{
	namespace engine
	{
		namespace activation
		{
			namespace cpu
			{
				class OneHotActivation : public ActivationFunction
				{
				public:
					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;

				protected:
					virtual neuron_value ComputeActivation(const neuron_value& value) = 0;
					virtual neuron_value ComputeDerivative(const neuron_value& value) = 0;
				};

				class SigmoidActivation : public OneHotActivation
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::sigmoid;
					}

				protected:
					neuron_value ComputeActivation(const neuron_value& value) override;
					neuron_value ComputeDerivative(const neuron_value& value) override;
				};

				// hyperbolic tangent activation.
				class TanhActivation : public OneHotActivation
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::tahn;
					}
				protected:
					neuron_value ComputeActivation(const neuron_value& value) override;
					neuron_value ComputeDerivative(const neuron_value& value) override;
				};

				class ReLuActivation : public OneHotActivation
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::reLU;
					}
				protected:
					virtual neuron_value ComputeActivation(const neuron_value& value) override;
					virtual neuron_value ComputeDerivative(const neuron_value& value) override;
				};

				class LeakyReLuActivation : public ReLuActivation
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::leakyReLU;
					}

				protected:
					neuron_value ComputeActivation(const neuron_value& value) override;
					neuron_value ComputeDerivative(const neuron_value& value) override;
				};

				class ELuActivation : public ReLuActivation
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::eLU;
					}

				protected:
					neuron_value ComputeActivation(const neuron_value& value) override;
					neuron_value ComputeDerivative(const neuron_value& value) override;
				};

				class SoftmaxActivation : public ActivationFunction
				{
				public:
					SoftmaxActivation();
					virtual ~SoftmaxActivation();

					virtual network::_activation_type GetType() const override{
						return network::_activation_type::softmax;
					}

				protected:

					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;

				private:
					_NEURO_TENSOR_DATA m_add_buf;
				};
			}
		}
	}
}

#endif
