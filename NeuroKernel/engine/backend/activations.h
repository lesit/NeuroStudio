#ifndef _ACTIVATION_FUNCTION_H
#define _ACTIVATION_FUNCTION_H

#include "common.h"
#include "../../network/NeuralNetworkTypes.h"

namespace np
{
	namespace engine
	{
		namespace activation
		{
			class ActivationFunction
			{
			public:
				ActivationFunction()
				{
				}
				virtual ~ActivationFunction(){}

				static ActivationFunction* CreateInstanceCPU(network::_activation_type type);
				static ActivationFunction* CreateInstanceCUDA(network::_activation_type type);

				virtual bool ForwardActivations(const _NEURO_TENSOR_DATA& value) = 0;
				virtual bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) = 0;

				virtual network::_activation_type GetType() const = 0;
				const wchar_t* GetTypeString() const
				{
					switch (GetType())
					{
						case network::_activation_type::sigmoid:
							return L"sigmoid";
						case network::_activation_type::tahn:
							return L"tahn";
						case network::_activation_type::reLU:
							return L"reLU";
						case network::_activation_type::leakyReLU:
							return L"leakyReLU";
						case network::_activation_type::eLU:
							return L"eLU";
						case network::_activation_type::softmax:
							return L"softmax";
					}
					return L"unknown";
				}

				neuron_value GetScaleMin() const 
				{
					switch (GetType())
					{
					case network::_activation_type::tahn:
						return neuron_value(-0.9);
					}
					return neuron_value(0.0); 
				}
				neuron_value GetScaleMax() const
				{
					switch (GetType())
					{
					case network::_activation_type::reLU:
					case network::_activation_type::leakyReLU:
					case network::_activation_type::eLU:
						return -1;	// min보다 작으니까 무한대라는 뜻
					case network::_activation_type::softmax:
						return 1.0;
					}
					return neuron_value(1.0);
				}

				static neuron_value m_relu_negative_slope;
				static neuron_value m_elu_alpha;
			};
		}
	}
}

#endif
