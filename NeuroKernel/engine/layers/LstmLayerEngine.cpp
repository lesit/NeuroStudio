#include "stdafx.h"

#include "LstmLayerEngine.h"
#include "LstmLayerCpuEngine.h"
#include "LstmLayerCudaEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

LstmLayerEngine* LstmLayerEngine::CreateInstance(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const RecurrentLayerEngine* prev_conn)
{
	if (net_param.run_pdtype == core::math_device_type::cpu)
		return new LstmLayerCpuEngine(net_param, layer, prev_conn);
	else
		return new LstmLayerCudaEngine(net_param, layer, prev_conn);
}

LstmLayerEngine::LstmLayerEngine(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const RecurrentLayerEngine* prev_conn)
: RecurrentLayerEngine(net_param, layer, prev_conn)
, m_last_time_cell(net_param.run_pdtype, true)
, m_last_time_cell_diff(net_param.run_pdtype, true)
, m_last_time_hidden(net_param.run_pdtype, true)
, m_last_time_hidden_diff(net_param.run_pdtype, true)
, m_prev_gate_diff(net_param.run_pdtype, true)
, m_gate(net_param.run_pdtype, true)
, m_cell(net_param.run_pdtype, true)
, m_cell_diff1(net_param.run_pdtype, true)
, m_cell_diff2(net_param.run_pdtype, true)
{
	m_clipping_threshold = 0;
}

LstmLayerEngine::~LstmLayerEngine()
{
}

neuro_u32 LstmLayerEngine::Get1MultiplierSizePerBatch() const
{
	return m_time_length;
}

bool LstmLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	// buf.value_size 는 time이 포함된 hidden size. 즉, T X hidden size
	if (m_last_time_cell.Calloc(buf.batch_size, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc last time cell buffer");
		return false;
	}

	if(m_last_time_cell_diff.Calloc(buf.batch_size, m_hidden_size)==NULL)
	{
		DEBUG_OUTPUT(L"failed alloc last time cell diff buffer");
		return false;
	}

	if (m_last_time_hidden.Calloc(buf.batch_size, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc last time hidden buffer");
		return false;
	}

	if (m_last_time_hidden_diff.Calloc(buf.batch_size, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc last time hidden diff buffer");
		return false;
	}

	if (m_cell.Calloc(buf.batch_size, m_time_length, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc cell buffer");
		return false;
	}

	if (m_cell_diff1.Calloc(buf.batch_size, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc cell diff 0 buffer");
		return false;
	}

	if (m_cell_diff2.Calloc(buf.batch_size, m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc cell diff 1 buffer");
		return false;
	}

	if (m_gate.Calloc(buf.batch_size, m_time_length, 4 * m_hidden_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc gate buffer");
		return false;
	}

	if (m_prev_gate_diff.Calloc(buf.batch_size, m_time_length, 4 * m_hidden_size) == NULL)// T * batch size X 4 X hidden size
	{
		DEBUG_OUTPUT(L"failed alloc previous gate diff buffer");
		return false;
	}

	return true;
}
