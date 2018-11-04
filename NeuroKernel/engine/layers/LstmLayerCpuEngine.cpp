#include "stdafx.h"

#include "LstmLayerCpuEngine.h"

#include "util/cpu_parallel_for.h"

using namespace np::engine;
using namespace np::engine::layers;

LstmLayerCpuEngine::LstmLayerCpuEngine(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const RecurrentLayerEngine* prev_conn)
: LstmLayerEngine(net_param, layer, prev_conn)
, m_hidden_to_gate(net_param.run_pdtype, true)
, m_hidden_to_hidden(net_param.run_pdtype, true)
{


}

LstmLayerCpuEngine::~LstmLayerCpuEngine()
{
}

bool LstmLayerCpuEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	if (!m_hidden_to_gate.Alloc(m_batch_total_gate_per_time))
	{
		DEBUG_OUTPUT(L"failed alloc m_hidden_to_gate GetBuffer()(%u)", m_batch_total_gate_per_time*sizeof(neuron_value));
		return false;
	}

	if (!m_hidden_to_hidden.Alloc(m_batch_hidden_per_time))
	{
		DEBUG_OUTPUT(L"failed alloc m_hidden_to_hidden GetBuffer()(%u)", m_batch_hidden_per_time*sizeof(neuron_value));
		return false;
	}
	return true;
}

inline neuron_value sigmoid(neuron_value x) 
{
	return neuron_value(1) / (neuron_value(1) + exp(-x));
}

bool LstmLayerCpuEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	const neuro_u32 batch_size = input_data.batch_size;
	const neuro_u32 input_size = input_data.value_size;

	const neuron_weight* weight = m_inner_data_vector[0].data.buffer;
	const neuron_weight* bias = m_inner_data_vector[1].data.buffer;
	const neuron_weight* hidden_weight_buf = m_inner_data_vector[2].data.buffer;

	const _NEURO_TENSOR_DATA* prev_time_cell_data = NULL;
	const _NEURO_TENSOR_DATA* prev_time_hidden_data = NULL;
	if (m_prev_rnn)
	{
		prev_time_cell_data = &((LstmLayerEngine*)m_prev_rnn)->GetLastCellData();
		prev_time_hidden_data = &((LstmLayerEngine*)m_prev_rnn)->GetLastHiddenData();
	}
	else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)	// 뒤 연결이 있으면 어떻게 해야하나?? ㅜㅜ
	{
		prev_time_cell_data = &m_last_time_cell;
		prev_time_hidden_data = &m_last_time_hidden;
	}

	if (!m_net_param.math.gemm(CblasNoTrans, CblasTrans
		, m_time_length*batch_size, m_total_gate_size, input_size
		, neuron_value(1.), input_data.GetBuffer(), weight
		, neuron_value(0.), m_gate.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed Compute input to hidden forward propagation 1");
		return false;
	}

	if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans
		, m_time_length * batch_size
		, m_total_gate_size
		, 1
		, neuron_value(1.), m_net_param.sdb.one_set_vector.GetBuffer(), bias
		, neuron_value(1.), m_gate.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed Compute input to hidden forward propagation 2");
		return false;
	}

	neuron_value* c_t = m_cell.GetBuffer();
	neuron_value* h_t = output_data.GetBuffer();
	neuron_value* gate_t = m_gate.GetBuffer();
	neuron_value* prev_c_t = prev_time_cell_data ? prev_time_cell_data->GetBuffer() : NULL;
	neuron_value* prev_h_t = prev_time_hidden_data ? prev_time_hidden_data->GetBuffer() : NULL;

	bool* cont_t = m_cont_idc ? m_cont_idc->GetBuffer() : NULL;
	for (neuro_u32 t = 0; t < m_time_length; t++)
	{
		// Hidden-to-hidden propagation
		if (prev_h_t)
		{
			if (!m_net_param.math.gemm(CblasNoTrans, CblasTrans
				, batch_size
				, m_total_gate_size
				, m_hidden_size
				, neuron_value(1.), prev_h_t, hidden_weight_buf
				, neuron_value(0.), m_hidden_to_gate.GetBuffer()))
			{
				DEBUG_OUTPUT(L"failed hidden to hidden");
				return false;
			}
			for (int index = 0; index < m_batch_total_gate_per_time; index++)
			{
				const int sample = index / m_total_gate_size;

				bool is_cont = cont_t ? cont_t[sample] : t > 0;
				if (is_cont)
					gate_t[index] += m_hidden_to_gate.GetBuffer()[index];
			}
		}

		neuron_value* s_c_t=c_t;
		neuron_value* s_h_t=h_t;

		neuron_value* h_to_g = m_hidden_to_gate.GetBuffer();
		for (neuro_u32 sample = 0; sample < batch_size; sample++)
		{
			bool is_cont = cont_t ? cont_t[sample] : t > 0;
			/*
			if (prev_h_t && is_cont)
			{
				for (neuro_size_t index = 0; index < m_total_gate_size; index++)
					gate_t[index] += h_to_g[index];
			}
			*/
			// hidden unit 개수 만큼 계산. 즉, LSTM layer에 속한 LSTM unit의 개수만큼
			for (int d = 0; d < m_hidden_size; ++d) 
			{	
				// Apply nonlinearity
				// i(t) : input gate
				gate_t[d] = sigmoid(gate_t[d]);									
				// f(t) : forget gate
				gate_t[m_hidden_size + d] = is_cont ? sigmoid(gate_t[m_hidden_size + d]) : 0.f;
				// o(t)
				gate_t[2 * m_hidden_size + d] = sigmoid(gate_t[2 * m_hidden_size + d]);
				// C`(t)
				gate_t[3 * m_hidden_size + d] = tanh(gate_t[3 * m_hidden_size + d]);

				// Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
				s_c_t[d] = gate_t[d] * gate_t[3 * m_hidden_size + d];
				if (prev_c_t != NULL && is_cont)
					s_c_t[d] += gate_t[m_hidden_size + d] * prev_c_t[d];

				// hidden state : o(t) * c(t)
				s_h_t[d] = gate_t[2 * m_hidden_size + d] * tanh(s_c_t[d]);
			}

			s_c_t += m_hidden_size;
			if (prev_c_t)
				prev_c_t += m_hidden_size;
			else
				int a = 0;
			s_h_t += m_hidden_size;

			gate_t += m_total_gate_size;
			h_to_g += m_total_gate_size;
		}

		prev_c_t = c_t;
		prev_h_t = h_t;

		c_t += m_batch_hidden_per_time;
		h_t += m_batch_hidden_per_time;

		if (cont_t)
			cont_t += batch_size;
	}

	if (prev_c_t)
	{
		if (!m_last_time_cell.CopyFrom(prev_c_t, m_last_time_cell.GetSize()))
		{
			DEBUG_OUTPUT(L"failed copy last cell data");
			return false;
		}
	}
	else
		m_last_time_cell.SetZero();

	if (prev_h_t)
	{
		if (!m_last_time_hidden.CopyFrom(prev_h_t, m_last_time_hidden.GetSize()))
		{
			DEBUG_OUTPUT(L"failed copy last hidden data");
			return false;
		}
	}
	else
		m_last_time_hidden.SetZero();

	m_last_time_cell_diff.SetZero();
	m_last_time_hidden_diff.SetZero();
	return true;
}

bool LstmLayerCpuEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	const neuro_u32 batch_size = output_data.batch_size;

	// 4개의 gate 전 값 계산
	const neuron_weight* weight_buf = m_inner_data_vector[0].data.GetBuffer();

	const neuron_weight* hidden_weight_buf = m_inner_data_vector[2].data.GetBuffer();

	const neuron_value* c_t = m_cell.GetBuffer() + (m_time_length - 1) * m_batch_hidden_per_time;
	neuron_value* gate_t = m_gate.GetBuffer() + (m_time_length - 1) * m_batch_total_gate_per_time;

	neuron_value* dh_t = current_error.GetBuffer() + (m_time_length - 1) * m_batch_hidden_per_time;
	neuron_value* pre_gate_diff_t = m_prev_gate_diff.GetBuffer() + (m_time_length - 1) * m_batch_total_gate_per_time;

	// 다음 lstm layer에서 전달한 error를 반영한다.
	GetCellDiff(m_time_length - 1).CopyFrom(m_last_time_cell_diff);
	GetCellDiff(m_time_length).SetZero();

	// 다음 lstm layer에서 전달한 hidden error를 반영한다.
	if (!current_error.data.mm.Memcpy(dh_t, m_last_time_hidden_diff.GetBuffer(), m_last_time_hidden_diff.GetSize(), current_error.data.mm))
	{
		DEBUG_OUTPUT(L"failed memcpy");
		return false;
	}

	const _NEURO_TENSOR_DATA* prev_time_cell_data = NULL;
	const _NEURO_TENSOR_DATA* prev_time_cell_diff = NULL;
	const _NEURO_TENSOR_DATA* prev_time_hidden_diff = NULL;

	if (m_prev_rnn)	// 여기에서 이전 lstm으로 error를 전달해주기 때문에 다음 lstm의 error를 가져올 필요가 없다.
	{
		prev_time_cell_data = &((LstmLayerEngine*)m_prev_rnn)->GetLastCellData();
		prev_time_cell_diff = &((LstmLayerEngine*)m_prev_rnn)->GetLastCellDiff();
		prev_time_hidden_diff = &((LstmLayerEngine*)m_prev_rnn)->GetLastHiddenDiff();
	}
	// 요건 앞뒤 연결이 없을때 하는 건데... 좀더 고민해봐야 함!
	else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)
	{
		prev_time_cell_data = &m_last_time_cell;
		prev_time_cell_diff = &m_last_time_cell_diff;
		prev_time_hidden_diff = &m_last_time_hidden_diff;
	}

	bool* cont_t = m_cont_idc ? m_cont_idc->GetBuffer() + (m_time_length - 1) * batch_size : NULL;

	for (int t = m_time_length - 1; t >= 0; --t)
	{
		const neuron_value* c_prev = t > 0 ? c_t - m_batch_hidden_per_time : (prev_time_cell_data ? prev_time_cell_data->GetBuffer() : NULL);

		neuron_value* dh_prev = t > 0 ? dh_t - m_batch_hidden_per_time : (prev_time_hidden_diff ? prev_time_hidden_diff->GetBuffer() : NULL);
		neuron_value* dc_prev = t > 0 ? GetCellDiff(t-1).GetBuffer() : (prev_time_cell_diff ? prev_time_cell_diff->GetBuffer() : NULL);

		neuron_value* dc_t = GetCellDiff(t).GetBuffer();

		neuron_value* s_dh_t = dh_t;	// 두번째 dh_t 값 이상
		const neuron_value* s_c_t = c_t;
		const neuron_value* s_c_prev = c_prev;
		neuron_value* s_dc_t = dc_t;
		neuron_value* s_dc_prev = dc_prev;
		neuron_value* s_gate_t = gate_t;
		neuron_value* s_pre_gate_diff_t = pre_gate_diff_t;

		for (neuro_u32 sample = 0; sample < batch_size; sample++)
		{
			const bool is_cont = cont_t ? cont_t[sample] : t > 0;

			for (int d = 0; d < m_hidden_size; ++d) 
			{
				const neuron_value tanh_c = tanh(s_c_t[d]);	// h_t가 만들어질때 사용된 c_t의 tanh 값
				
				s_dc_t[d] += s_dh_t[d] * s_gate_t[2 * m_hidden_size + d] * (neuron_value(1) - tanh_c * tanh_c);

				if (s_dc_prev)
					s_dc_prev[d] = is_cont ? s_dc_t[d] * s_gate_t[m_hidden_size + d] : neuron_value(0);

				s_pre_gate_diff_t[d] = s_dc_t[d] * s_gate_t[3 * m_hidden_size + d];
				s_pre_gate_diff_t[m_hidden_size + d] = is_cont && s_c_prev ? s_dc_t[d] * s_c_prev[d] : neuron_value(0);
				s_pre_gate_diff_t[2 * m_hidden_size + d] = s_dh_t[d] * tanh_c;
				s_pre_gate_diff_t[3 * m_hidden_size + d] = s_dc_t[d] * s_gate_t[d];

				s_pre_gate_diff_t[d] *= s_gate_t[d] * (neuron_value(1) - s_gate_t[d]);
				s_pre_gate_diff_t[m_hidden_size + d] *= s_gate_t[m_hidden_size + d]	* (1 - s_gate_t[m_hidden_size + d]);
				s_pre_gate_diff_t[2 * m_hidden_size + d] *= s_gate_t[2 * m_hidden_size + d]	* (1 - s_gate_t[2 * m_hidden_size + d]);
				s_pre_gate_diff_t[3 * m_hidden_size + d] *= (neuron_value(1) - s_gate_t[3 * m_hidden_size + d] * s_gate_t[3 * m_hidden_size + d]);
			}

			// Clip deriviates before nonlinearity
			if (m_clipping_threshold > 0.) 
			{
				for (neuro_size_t index = 0; index < m_total_gate_size; index++)
				{
					if (s_pre_gate_diff_t[index] < -m_clipping_threshold)
						s_pre_gate_diff_t[index] = -m_clipping_threshold;
					else if (s_pre_gate_diff_t[index] > m_clipping_threshold)
						s_pre_gate_diff_t[index] = m_clipping_threshold;
				}
			}

			s_dh_t += m_hidden_size;
			s_c_t += m_hidden_size;

			if (s_c_prev)
				s_c_prev += m_hidden_size;

			s_dc_t += m_hidden_size;

			if (s_dc_prev)
				s_dc_prev += m_hidden_size;

			s_gate_t += m_total_gate_size;
			s_pre_gate_diff_t += m_total_gate_size;
		}

		// Backprop errors to the previous time step
		if (dh_prev)
		{
			if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans
				, batch_size
				, m_hidden_size
				, m_total_gate_size
				, neuron_value(1.), pre_gate_diff_t, hidden_weight_buf
				, neuron_value(0.), m_hidden_to_hidden.GetBuffer()))
			{
				DEBUG_OUTPUT(L"failed hidden to prev gate");
				return false;
			}
			neuron_value* h_to_h = m_hidden_to_hidden.GetBuffer();
			for (neuro_u32 sample = 0; sample < batch_size; sample++)
			{
				const bool is_cont = cont_t ? cont_t[sample] : t > 0;
				if (is_cont)
				{
					for (neuro_size_t index = 0; index < m_hidden_size; index++)
						dh_prev[index] += h_to_h[index];
				}
				h_to_h += m_hidden_size;
				dh_prev += m_hidden_size;
			}
		}

		c_t -= m_batch_hidden_per_time;
		gate_t -= m_batch_total_gate_per_time;
		dh_t -= m_batch_hidden_per_time;
		pre_gate_diff_t -= m_batch_total_gate_per_time;
		if (cont_t)
			cont_t -= batch_size;
	}

	if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans
		, m_time_length * batch_size
		, input_error.value_size
		, m_total_gate_size
		, neuron_value(1.), m_prev_gate_diff.GetBuffer(), weight_buf
		, neuron_value(0.), input_error.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed hidden to prev gate");
		return false;
	}
	return true;
}

bool LstmLayerCpuEngine::BackwardWeight(neuro_u32 index
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& input_data
	, const _VALUE_VECTOR& grad_weight)
{
	if (index == 0)
	{
		const neuro_u32 batch_size = input_data.batch_size;
		const neuro_u32 input_size = input_data.value_size;

		if (grad_weight.count != input_size * m_total_gate_size)
		{
			DEBUG_OUTPUT(L"grad weight's size is strange");
			return false;
		}

		// Gradient w.r.t. input-to-hidden weight
		if (!m_net_param.math.gemm(CblasTrans, CblasNoTrans
			, m_total_gate_size
			, input_size
			, m_time_length * batch_size
			, neuron_value(1.), m_prev_gate_diff.GetBuffer(), input_data.GetBuffer()
			, neuron_value(0.), grad_weight.GetBuffer()))
		{
			DEBUG_OUTPUT(L"failed hidden to prev gate");
			return false;
		}
	}
	else if(index==1)
	{
		if (grad_weight.count != m_total_gate_size)
		{
			DEBUG_OUTPUT(L"grad bias's size is strange");
			return false;
		}

		for (neuro_u32 i = 0, n = m_time_length * current_error.batch_size*m_total_gate_size; i < n; i++)
		{
			if (_finite(m_prev_gate_diff.GetBuffer()[i]) == 0)
				int a = 0;
		}

		cblas_sgemv(CblasRowMajor, CblasTrans
			, m_time_length * current_error.batch_size
			, m_total_gate_size
			, 1.
			, m_prev_gate_diff.GetBuffer(), m_total_gate_size
			, m_net_param.sdb.one_set_vector.GetBuffer(), 1
			, 0.
			, grad_weight.GetBuffer(), 1);
	}
	else
	{
		if (grad_weight.count != m_hidden_size * m_total_gate_size)
		{
			DEBUG_OUTPUT(L"grad weight's size is strange");
			return false;
		}

		const _NEURO_TENSOR_DATA* prev_time_hidden_data = NULL;
		if (m_prev_rnn)
		{
			prev_time_hidden_data = &((LstmLayerEngine*)m_prev_rnn)->GetLastHiddenData();
		}
		else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)
		{
			prev_time_hidden_data = &m_last_time_hidden;
		}

		// Gradient w.r.t. hidden-to-hidden weight
		// Add Gradient from previous time-step
		if (!m_net_param.math.gemm(CblasTrans, CblasNoTrans
			, m_total_gate_size
			, m_hidden_size
			, (m_time_length - 1)*output_data.batch_size
			, neuron_value(1.), m_prev_gate_diff.GetBuffer() + m_batch_total_gate_per_time, output_data.GetBuffer()
			, neuron_value(0.), grad_weight.GetBuffer()))
		{
			DEBUG_OUTPUT(L"Gradient w.r.t. hidden-to-hidden weight");
			return false;
		}
		if (prev_time_hidden_data)
		{
			if (!m_net_param.math.gemm(CblasTrans, CblasNoTrans
				, m_total_gate_size
				, m_hidden_size
				, 1
				, neuron_value(1.), m_prev_gate_diff.GetBuffer(), prev_time_hidden_data->GetBuffer()
				, neuron_value(1.), grad_weight.GetBuffer()))
			{
				DEBUG_OUTPUT(L"Gradient w.r.t. hidden-to-hidden weight");
				return false;
			}
		}
	}

	return true;
}
