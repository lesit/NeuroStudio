#include "ConvLayerCudnnEngine.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;
using namespace np::core::cuda;

#define CUDNN_STREAMS_PER_GROUP 3	// weight하고 bias의 gradient 계산을 위해 두개를 추가한 3개이다.

ConvLayerCudnnEngine::ConvLayerCudnnEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: ConvLayerEngineBase(net_param, layer)
{
	// initialize all to default algorithms
	m_fwd_algo = (cudnnConvolutionFwdAlgo_t)0;
	m_bwd_filter_algo = (cudnnConvolutionBwdFilterAlgo_t)0;
	m_bwd_data_algo = (cudnnConvolutionBwdDataAlgo_t)0;

	m_input_desc = NULL;
	m_output_desc = NULL;

	m_bias_desc = NULL;
	m_filter_desc = NULL;
	m_conv_desc = NULL;

	// workspace data
	m_workspaceData = NULL;
}

ConvLayerCudnnEngine::~ConvLayerCudnnEngine()
{
	if (m_input_desc)
		cudnnDestroyTensorDescriptor(m_input_desc);
	if (m_output_desc)
		cudnnDestroyTensorDescriptor(m_output_desc);

	if (m_bias_desc)
		cudnnDestroyTensorDescriptor(m_bias_desc);

	if (m_filter_desc)
		cudnnDestroyFilterDescriptor(m_filter_desc);
	if (m_conv_desc)
		cudnnDestroyConvolutionDescriptor(m_conv_desc);

	for (size_t i = 0; i < m_cuda_handle_vector.size(); i++)
	{
		_PARALLEL_INFO& info = m_cuda_handle_vector[i];
		if (info.stream)
			cudaStreamDestroy(info.stream);

		if (info.handle)
			cudnnDestroy(info.handle);
	}

	if (m_workspaceData)
		cudaFree(m_workspaceData);
}

bool ConvLayerCudnnEngine::OnInitialized()
{
	if (!__super::OnInitialized())
	{
		DEBUG_OUTPUT(L"failed super class initialize");
		return false;
	}

	// workspace data
	m_workspaceSizeInBytes = 0;
	m_workspaceData = NULL;

	// initialize all to default algorithms
	m_fwd_algo = (cudnnConvolutionFwdAlgo_t)0;
	m_bwd_filter_algo = (cudnnConvolutionBwdFilterAlgo_t)0;
	m_bwd_data_algo = (cudnnConvolutionBwdDataAlgo_t)0;

	// default algorithms don't require workspace
	m_workspace_fwd_size = 0;
	m_workspace_bwd_data_size = 0;
	m_workspace_bwd_filter_size = 0;

	for (int stream = 0; stream < CUDNN_STREAMS_PER_GROUP; stream++) 
	{
		_PARALLEL_INFO info;
		if (!CudaPlatform::CudaErrorCheck(cudaStreamCreate(&info.stream)))
			return false;

		if (!CUDNN_CHECK(cudnnCreate(&info.handle)))
			return false;

		if (!CUDNN_CHECK(cudnnSetStream(info.handle, info.stream)))
			return false;

		m_cuda_handle_vector.push_back(info);
	}

	if (!CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc)))
		return false;
	if (!CUDNN_CHECK(cudnnSetFilter4dDescriptor(m_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW
		, m_out_ts.GetChannelCount(), m_in_ts.GetChannelCount(), m_entry.conv.filter.kernel_height, m_entry.conv.filter.kernel_width)))
		return false;

	// Create tensor descriptor(s) for data and corresponding convolution(s).
	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc)))
		return false;
	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc)))
		return false;

	if (m_inner_data_vector[1].data.count>0)
	{
		// Tensor descriptor for bias.
		if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_bias_desc)))
			return false;
	}

	if (!CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc)))
		return false;

	return true;
}

bool ConvLayerCudnnEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	// Specify workspace limit for kernels directly until we have a
	// planning strategy and a rewrite of Caffe's GPU memory mangagement
	size_t workspace_limit_bytes = 8 * 1024 * 1024;

	Timer timer;
	if (!CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT
		, buf.GetBatchTimeSize()
		, m_in_ts.GetChannelCount()
		, m_in_ts.GetHeight()
		, m_in_ts.GetWidth())))
	{
		DEBUG_OUTPUT(L"failed cudnnSetTensor4dDescriptor for input. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	if (!CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT
		, buf.GetBatchTimeSize()
		, m_out_ts.GetChannelCount()
		, m_out_ts.GetHeight()
		, m_out_ts.GetWidth())))
	{
		DEBUG_OUTPUT(L"failed cudnnSetTensor4dDescriptor for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	neuro_float elapse = timer.elapsed();

	// Tensor descriptor for bias.
	if (m_bias_desc != NULL)
	{
		if (!CUDNN_CHECK(cudnnSetTensor4dDescriptor(m_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT
			, 1
			, m_out_ts.GetChannelCount()
			, 1
			, 1)))
		{
			DEBUG_OUTPUT(L"failed cudnnSetTensor4dDescriptor for bias. %s", CudaPlatform::GetErrorString().c_str());
			return false;
		}
	}

	if (!CUDNN_CHECK(cudnnSetConvolution2dDescriptor(m_conv_desc
		, m_pad_height.first, m_pad_width.first
		, m_entry.conv.filter.stride_height, m_entry.conv.filter.stride_width
		, m_entry.conv.dilation_height, m_entry.conv.dilation_width, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)))
	{
		DEBUG_OUTPUT(L"failed cudnnSetConvolution2dDescriptor for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	if (!CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(m_cuda_handle_vector[0].handle,
		m_input_desc,
		m_filter_desc,
		m_conv_desc,
		m_output_desc,
		CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes,
		&m_fwd_algo)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionForwardAlgorithm for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	if (!CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(m_cuda_handle_vector[0].handle,
		m_input_desc,
		m_filter_desc,
		m_conv_desc,
		m_output_desc,
		m_fwd_algo,
		&m_workspace_fwd_size)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionForwardAlgorithm for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	// choose backward algorithm for filter
	if (!CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(m_cuda_handle_vector[0].handle,
		m_input_desc,
		m_output_desc,
		m_conv_desc,
		m_filter_desc,
		CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &m_bwd_filter_algo)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionBackwardFilterAlgorithm for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	// get workspace for backwards filter algorithm
	if (!CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_cuda_handle_vector[0].handle,
		m_input_desc, m_output_desc, m_conv_desc, m_filter_desc,
		m_bwd_filter_algo, &m_workspace_bwd_filter_size)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionBackwardFilterWorkspaceSize for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	// choose backward algo for data
	if (!CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(m_cuda_handle_vector[0].handle,
		m_filter_desc, m_output_desc, m_conv_desc, m_input_desc,
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &m_bwd_data_algo)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionBackwardFilterWorkspaceSize for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	// get workspace size
	if (!CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(m_cuda_handle_vector[0].handle,
		m_filter_desc, m_output_desc, m_conv_desc, m_input_desc,
		m_bwd_data_algo, &m_workspace_bwd_data_size)))
	{
		DEBUG_OUTPUT(L"failed cudnnGetConvolutionBackwardDataWorkspaceSize for output. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}


	// get max over all operations
	size_t max_workspace = max(m_workspace_fwd_size, m_workspace_bwd_data_size);
	max_workspace = max(max_workspace, m_workspace_bwd_filter_size);
	// ensure all groups have enough workspace
	size_t total_max_workspace = max_workspace * m_cuda_handle_vector.size();

	// this is the total amount of storage needed over all groups + streams
	if (total_max_workspace > m_workspaceSizeInBytes)
	{
		DEBUG_OUTPUT(L"Reallocating workspace storage: %llu", total_max_workspace);
		m_workspaceSizeInBytes = total_max_workspace;

		// free the existing workspace and allocate a new (larger) one
		if (m_workspaceData)
			cudaFree(m_workspaceData);

		cudaError_t err = cudaMalloc(&m_workspaceData, m_workspaceSizeInBytes);
		// if we succeed in the allocation, set pointer aliases for workspaces
		if (err == cudaSuccess)
		{
			for (int stream = 0; stream < m_cuda_handle_vector.size(); stream++)
				m_cuda_handle_vector[stream].workspace = reinterpret_cast<char *>(m_workspaceData)+max_workspace;
		}
		else
		{
			// force zero memory path
			m_workspace_fwd_size = 0;
			m_workspace_bwd_filter_size = 0;
			m_workspace_bwd_data_size = 0;
			m_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			m_bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			m_bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

			// NULL out all workspace pointers
			for (int stream = 0; stream < m_cuda_handle_vector.size(); stream++)
				m_cuda_handle_vector[stream].workspace = NULL;

			// NULL out underlying data
			m_workspaceData = NULL;
			m_workspaceSizeInBytes = 0;
		}
	}

	return true;
}

__global__ void sync_conv_groups() { }

bool ConvLayerCudnnEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	// Filters.
	if (!CUDNN_CHECK(cudnnConvolutionForward(m_cuda_handle_vector[0].handle
		, dataType::one
		, m_input_desc, input_data.GetBuffer()
		, m_filter_desc, m_inner_data_vector[0].data.GetBuffer()
		, m_conv_desc
		, m_fwd_algo, m_cuda_handle_vector[0].workspace, m_workspace_fwd_size
		, dataType::zero
		, m_output_desc, output_data.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnConvolutionForward. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	sync_conv_groups << <1, 1 >> >();

	if (m_inner_data_vector[1].data.GetBuffer())
	{
		if (!CUDNN_CHECK(cudnnAddTensor(m_cuda_handle_vector[0].handle
			, dataType::one
			, m_bias_desc, m_inner_data_vector[1].data.GetBuffer()
			, dataType::one
			, m_output_desc, output_data.GetBuffer())))
		{
			DEBUG_OUTPUT(L"failed cudnnAddTensor for bias. %s", CudaPlatform::GetErrorString().c_str());
			return false;
		}

		sync_conv_groups << <1, 1 >> >();
	}
	return true;
}

bool ConvLayerCudnnEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	// input error
	if (!CUDNN_CHECK(cudnnConvolutionBackwardData(
		m_cuda_handle_vector[2].handle,
		dataType::one,
		m_filter_desc, m_inner_data_vector[0].data.GetBuffer(),
		m_output_desc, current_error.GetBuffer(),
		m_conv_desc,
		m_bwd_data_algo, m_cuda_handle_vector[2].workspace,
		m_workspace_bwd_data_size,
		dataType::zero,
		m_input_desc, input_error.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnConvolutionBackwardData. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	sync_conv_groups << <1, 1 >> >();
	return true;
}

bool ConvLayerCudnnEngine::BackwardWeight(neuro_u32 index
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& input_data
	, const _VALUE_VECTOR& grad_weight)
{
	if (index == 0)
	{
		// 정확히 cudnnConvolutionBackwardFilter 만 문제 있음!!!
		// Gradient w.r.t. weights.
		if (!CUDNN_CHECK(cudnnConvolutionBackwardFilter(
			m_cuda_handle_vector[1].handle,
			dataType::one,
			m_input_desc, input_data.GetBuffer(),
			m_output_desc, current_error.GetBuffer(),
			m_conv_desc,
			m_bwd_filter_algo, m_cuda_handle_vector[1].workspace,
			m_workspace_bwd_filter_size,
			dataType::one,
			m_filter_desc, grad_weight.GetBuffer())))
		{
			DEBUG_OUTPUT(L"failed cudnnConvolutionBackwardFilter. %s", CudaPlatform::GetErrorString().c_str());
			return false;
		}
	}
	else
	{
		// Gradient w.r.t. bias.
		if (m_bias_desc == NULL)
			return true;

		// cudnn에서는 db_minibatch 가 필요없다. 왜냐면 한번에 merge까지 해주므로..

		if (!CUDNN_CHECK(cudnnConvolutionBackwardBias(m_cuda_handle_vector[0].handle,
			dataType::one,
			m_output_desc, current_error.GetBuffer(),
			dataType::one,
			m_bias_desc, grad_weight.GetBuffer())))
		{
			DEBUG_OUTPUT(L"failed cudnnConvolutionBackwardBias. %s", CudaPlatform::GetErrorString().c_str());
			return false;
		}
	}
	sync_conv_groups << <1, 1 >> >();
	return true;
}
