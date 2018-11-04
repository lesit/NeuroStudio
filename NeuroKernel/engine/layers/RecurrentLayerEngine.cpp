#include "stdafx.h"

#include "RecurrentLayerEngine.h"
#include "LstmLayerEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

RecurrentLayerEngine* RecurrentLayerEngine::CreateInstance(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const HiddenLayerEngine* side_layer_engine)
{
	if (side_layer_engine->GetLayerType() != network::_layer_type::rnn)
		side_layer_engine = NULL;

	if ((network::_rnn_type)layer.GetEntry().rnn.type==network::_rnn_type::lstm)
		return LstmLayerEngine::CreateInstance(net_param, layer, (RecurrentLayerEngine*)side_layer_engine);

	/*
	if (m_entry.rnn.is_non_time_input == 0 && GetInputLayers().size() > 0)
	m_time_length = m_input_vector[0].layer->GetOutTensorShape().time_length;
	else
	m_time_length = m_entry.rnn.fix_time_length;
	*/
	return NULL;
}

RecurrentLayerEngine::RecurrentLayerEngine(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const RecurrentLayerEngine* prev_conn)
: HiddenLayerEngine(net_param, layer)
{
	m_time_length = m_entry.rnn.fix_time_length;
	m_prev_rnn = prev_conn;

	m_cont_idc = NULL;
	// data 가 epoch 안에서 계속 연속적으로 이동한다면 m_cont_idc=NULL 일 경우에도 이전 hidden/cell 값을 사용 할 수 있다.
	m_bSequanceInEpoch = false;	

	m_hidden_size = 0;
	m_total_gate_size = 0;
}

RecurrentLayerEngine::~RecurrentLayerEngine()
{
}

bool RecurrentLayerEngine::OnInitialized()
{
	if (m_entry.rnn.is_non_time_input && GetInputLayers().size() == 0)
		return false;

	m_hidden_size = m_entry.rnn.output_count;

	switch ((network::_rnn_type)m_entry.rnn.type)
	{
	case network::_rnn_type::lstm:
		m_total_gate_size = 4 * m_hidden_size;
		break;
	case network::_rnn_type::gru:
		m_total_gate_size = 3 * m_hidden_size;
		break;
	}

	return true;
}

bool RecurrentLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	if (buf.time_length != m_time_length)
	{
		DEBUG_OUTPUT(L"output time length is different");
		return false;
	}

	if (m_cont_idc)
	{
		if (m_cont_idc->batch_size != buf.batch_size || m_cont_idc->time_length != buf.time_length)
			m_cont_idc = NULL;
	}

	m_batch_total_gate_per_time = buf.batch_size * m_total_gate_size;
	m_batch_hidden_per_time = buf.batch_size * m_hidden_size;
	return true;
}

