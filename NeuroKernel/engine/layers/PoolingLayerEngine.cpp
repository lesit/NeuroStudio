#include "stdafx.h"

#include "PoolingLayerEngine.h"
#include "PoolingLayerCudaEngine.h"
#include "PoolingLayerCpuEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

PoolingLayerEngine* PoolingLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	if (net_param.run_pdtype == core::math_device_type::cuda)
		return new PoolingLayerCudaEngine(net_param, layer);
	else
		return new PoolingLayerCpuEngine(net_param, layer);
}

PoolingLayerEngine::PoolingLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: HiddenLayerEngine(net_param, layer)
{
}


PoolingLayerEngine::~PoolingLayerEngine()
{
}

bool PoolingLayerEngine::OnInitialized()
{
	if (m_out_ts.GetHeight() != core::filter_output_length(m_in_ts.GetHeight(), m_entry.pooling.filter.kernel_height, m_entry.pooling.filter.stride_height))
	{
		DEBUG_OUTPUT(L"output height is strange");
		return false;
	}
	if (m_out_ts.GetWidth() != core::filter_output_length(m_in_ts.GetWidth(), m_entry.pooling.filter.kernel_width, m_entry.pooling.filter.stride_width))
	{
		DEBUG_OUTPUT(L"output width is strange");
		return false;
	}
	return true;
}
