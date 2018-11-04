#pragma once

#include "ConvLayerEngineBase.h"

#include "core/cuda_platform.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class ConvLayerCudnnEngine : public ConvLayerEngineBase
			{
			public:
				ConvLayerCudnnEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~ConvLayerCudnnEngine();

				virtual bool OnInitialized() override;

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;
			protected:
				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;

				bool BackwardWeight(neuro_u32 index
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& input_data
					, const _VALUE_VECTOR& grad_weight) override;

				// algorithms for forward and backwards convolutions
				cudnnConvolutionFwdAlgo_t m_fwd_algo;
				cudnnConvolutionBwdFilterAlgo_t m_bwd_filter_algo;
				cudnnConvolutionBwdDataAlgo_t m_bwd_data_algo;

				neuro_u32 m_batch_size;
				cudnnTensorDescriptor_t		m_input_desc;
				cudnnTensorDescriptor_t		m_output_desc;

				cudnnTensorDescriptor_t		m_bias_desc;

				cudnnFilterDescriptor_t		m_filter_desc;
				cudnnConvolutionDescriptor_t	m_conv_desc;

				size_t m_workspace_fwd_size;
				size_t m_workspace_bwd_data_size;
				size_t m_workspace_bwd_filter_size;

				size_t m_workspaceSizeInBytes;  // size of underlying storage
				void *m_workspaceData;  // underlying storage

				struct _PARALLEL_INFO
				{
					_PARALLEL_INFO()
					{
						handle = NULL;
						stream = NULL;
						workspace = NULL;
					}
					cudnnHandle_t handle;
					cudaStream_t  stream;
					void *workspace;// aliases into m_workspaceData
				};
				std::vector<_PARALLEL_INFO> m_cuda_handle_vector;
			};
		}
	}
}
