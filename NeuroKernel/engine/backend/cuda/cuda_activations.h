#ifndef _CUDA_ACTIVATION_FUNCTION_H
#define _CUDA_ACTIVATION_FUNCTION_H

#include "../activations.h"
#include "core/cuda_platform.h"

namespace np
{
	namespace engine
	{
		namespace activation
		{
			namespace cuda
			{
				class CuDNNActivation : public ActivationFunction
				{
				public:
					CuDNNActivation(network::_activation_type type);
					virtual ~CuDNNActivation();

					virtual network::_activation_type GetType() const override{
						return m_type;
					}

					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;
				protected:
					const network::_activation_type m_type;

					cudnnHandle_t m_cudnn_handle;
					cudnnActivationDescriptor_t m_activ_desc;
					cudnnTensorDescriptor_t m_ts_desc;
				};

				class CudaLeakyReLuActivation : public ActivationFunction
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::leakyReLU;
					}

					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;
				};

				class CudaELuActivation : public ActivationFunction
				{
				public:
					virtual network::_activation_type GetType() const override{
						return network::_activation_type::eLU;
					}

					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;
				};

				class CudaSoftmaxActivation : public ActivationFunction
				{
				public:
					CudaSoftmaxActivation();
					virtual ~CudaSoftmaxActivation();

					virtual network::_activation_type GetType() const override{
						return network::_activation_type::softmax;
					}

					bool ForwardActivations(const _NEURO_TENSOR_DATA& value) override;
					bool BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error) override;

				private:
					/*
					cudnnHandle_t m_cudnn_handle;
					cudnnTensorDescriptor_t m_in_desc;
					cudnnTensorDescriptor_t m_out_desc;
					*/
				};
			}
		}
	}
}

#endif
