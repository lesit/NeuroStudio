#include "PoolingLayerCudaEngine.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;
using namespace np::core::cuda;

PoolingLayerCudaEngine::PoolingLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: PoolingLayerEngine(net_param, layer)
{
	// initialize all to default algorithms
	m_handle = NULL;
	m_input_desc = NULL;
	m_output_desc = NULL;
	m_pooling_desc = NULL;
}

PoolingLayerCudaEngine::~PoolingLayerCudaEngine()
{
	if (m_input_desc)
		cudnnDestroyTensorDescriptor(m_input_desc);
	if (m_output_desc)
		cudnnDestroyTensorDescriptor(m_output_desc);
	if (m_pooling_desc)
		cudnnDestroyPoolingDescriptor(m_pooling_desc);
	if (m_handle)
		cudnnDestroy(m_handle);
}

bool PoolingLayerCudaEngine::OnInitialized()
{
	if (!__super::OnInitialized())
	{
		return false;
	}

	if (!CUDNN_CHECK(cudnnCreate(&m_handle)))
		return false;

	// Create tensor descriptor(s) for data and corresponding convolution(s).
	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc)))
		return false;
	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc)))
		return false;

	if (!CUDNN_CHECK(cudnnCreatePoolingDescriptor(&m_pooling_desc)))
		return false;

	cudnnPoolingMode_t mode;
	if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::ave_pooling)
		mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	else //if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::max_pooling)
		mode = CUDNN_POOLING_MAX;

	neuro_u16 pad_height = 0;
	neuro_u16 pad_width = 0;

	const nsas::_FILTER_ENTRY& filter = m_entry.pooling.filter;
	if (!CUDNN_CHECK(cudnnSetPooling2dDescriptor(m_pooling_desc, mode, CUDNN_PROPAGATE_NAN
		, filter.kernel_height, filter.kernel_width
		, pad_height, pad_width
		, filter.stride_height, filter.stride_width)))
	{
		DEBUG_OUTPUT(L"failed cudnnSetPooling2dDescriptor");
		return false;
	}

	return true;
}

bool PoolingLayerCudaEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	if (!CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT
		, buf.GetBatchTimeSize()
		, m_in_ts.GetChannelCount()
		, m_in_ts.GetHeight()
		, m_in_ts.GetWidth())))
	{
		DEBUG_OUTPUT(L"failed cudnnSetTensor4dDescriptor for input");
		return false;
	}

	if (!CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT
		, buf.GetBatchTimeSize()
		, m_out_ts.GetChannelCount()
		, m_out_ts.GetHeight()
		, m_out_ts.GetWidth())))
	{
		DEBUG_OUTPUT(L"failed cudnnSetTensor4dDescriptor for output");
		return false;
	}

	return true;
}

bool PoolingLayerCudaEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	if (!CUDNN_CHECK(cudnnPoolingForward(m_handle, m_pooling_desc,
		dataType::one,
		m_input_desc, input_data.GetBuffer(),
		dataType::zero,
		m_output_desc, output_data.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnPoolingForward");
		return false;
	}
	return true;
}

bool PoolingLayerCudaEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	if(!CUDNN_CHECK(cudnnPoolingBackward(m_handle, m_pooling_desc
			, dataType::one
			, m_output_desc, output_data.GetBuffer()
			, m_output_desc, current_error.GetBuffer()
			, m_input_desc, input_data.GetBuffer()
			, dataType::zero
			, m_input_desc, input_error.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnPoolingBackward");
		return false;
	}
	return true;
}
