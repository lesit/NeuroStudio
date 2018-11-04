#include "LstmLayerCudaEngine.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;
using namespace np::core::cuda;


LstmLayerCudaEngine::LstmLayerCudaEngine(const NetworkParameter& net_param
	, const network::HiddenLayer& layer
	, const RecurrentLayerEngine* prev_conn)
: LstmLayerEngine(net_param, layer, prev_conn)
{
}

LstmLayerCudaEngine::~LstmLayerCudaEngine()
{
}

__device__ neuron_value d_sigmoid(neuron_value x)
{
	return 1. / (1. + exp(-x));
}

// n = batch x 4 x hidden
__global__ void ClipAdd(const int n, const int dim, const bool* cont, const bool isNotFirst, const neuron_value* add_vec, neuron_value* data)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		const int sample = index / dim;
		if (cont && cont[sample] || cont == NULL && isNotFirst)
			data[index] += add_vec[index];
	}
}

__global__ void ActivationForward(const int n, const int hidden_size, neuron_value* gate)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		if (index % (4 * hidden_size) < 3 * hidden_size)
			gate[index] = d_sigmoid(gate[index]);
		else
			gate[index] = tanh(gate[index]);
	}
}

__global__ void LSTMForward(const int n, const int hidden_size
	, const bool* cont, const bool isNotFirst, const neuron_value* c_prev
	, const neuron_value* gate, neuron_value* c_t, neuron_value* h_t)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		const int sample = index / hidden_size;
		const int hidden_index = index % hidden_size;
		const neuron_value* gate_t = gate + 4 * hidden_size*sample;

		c_t[index] = gate_t[hidden_index] * gate_t[3 * hidden_size + hidden_index];
		if (c_prev != NULL && (cont ? cont[sample] : isNotFirst))
			c_t[index] += gate_t[hidden_size + hidden_index] * c_prev[index];

		h_t[index] = gate_t[2 * hidden_size + hidden_index] * tanh(c_t[index]);
	}
}

bool LstmLayerCudaEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	cublasHandle_t cublasHandle = m_net_param.cuda_instance->cublas_handle();
	if (cublasHandle == NULL)
	{
		DEBUG_OUTPUT(L"no cublas handle");
		return false;
	}

	const neuro_u32 batch_size = input_data.batch_size;
	const neuro_u32 input_size = input_data.value_size;

	const neuron_weight* weight = m_inner_data_vector[0].data.buffer;
	const neuron_weight* bias = m_inner_data_vector[1].data.buffer;
	const neuron_weight* hidden_weight = m_inner_data_vector[2].data.buffer;

	const _NEURO_TENSOR_DATA* prev_time_cell_data = NULL;
	const _NEURO_TENSOR_DATA* prev_time_hidden_data = NULL;
	if (m_prev_rnn)
	{
		prev_time_cell_data = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastCellData();
		prev_time_hidden_data = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastHiddenData();
	}
	else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)	// 뒤 연결이 있으면 어떻게 해야하나?? ㅜㅜ
	{
		prev_time_cell_data = &m_last_time_cell;
		prev_time_hidden_data = &m_last_time_hidden;
	}

	if (!CudaPlatform::CublasErrorCheck(
		cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N
		, m_total_gate_size
		, m_time_length*batch_size
		, input_size
		, &dataType::oneval
		, weight, input_size
		, input_data.GetBuffer(), input_size
		, &dataType::zeroval				// 여기에서 gate에 더하는게 아니라 값을 넣기 때문에 따로 전에 초기화할 필요가 없음
		, m_gate.GetBuffer(), m_total_gate_size)
		))
	{
		DEBUG_OUTPUT(L"failed cublasSgemm to calculation of pre gate for input weight. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	if (!CudaPlatform::CublasErrorCheck(
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N
		, m_total_gate_size
		, m_time_length * batch_size
		, 1
		, &dataType::oneval
		, bias, m_total_gate_size
		, m_net_param.sdb.one_set_vector.buffer, 1
		, &dataType::oneval
		, m_gate.GetBuffer(), m_total_gate_size)
		))
	{
		DEBUG_OUTPUT(L"failed cublasSgemm to calculation of pre gate for bias. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	neuron_value* c_t = m_cell.GetBuffer();
	neuron_value* h_t = output_data.GetBuffer();
	neuron_value* gate_t = m_gate.GetBuffer();
	const neuron_value* prev_c_t = prev_time_cell_data ? prev_time_cell_data->GetBuffer() : NULL;
	const neuron_value* prev_h_t = prev_time_hidden_data ? prev_time_hidden_data->GetBuffer() : NULL;

	_VALUE_VECTOR hidden_to_gate(m_net_param.run_pdtype, true);
	if (!hidden_to_gate.Alloc(m_batch_total_gate_per_time))
	{
		DEBUG_OUTPUT(L"failed alloc hidden_to_gate buffer(%u)", m_batch_total_gate_per_time*sizeof(neuron_value));
		return false;
	}

	bool* cont_t = m_cont_idc ? m_cont_idc->GetBuffer() : NULL;
	for (neuro_u32 t = 0; t < m_time_length; t++)
	{
		// Hidden-to-hidden propagation
		if (prev_h_t)
		{
			if (!CudaPlatform::CublasErrorCheck(
				cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N
				, m_total_gate_size
				, batch_size
				, m_hidden_size
				, &dataType::oneval
				, hidden_weight, m_hidden_size
				, prev_h_t, m_hidden_size
				, &dataType::zeroval				// 여기에서 gate에 더하는게 아니라 값을 넣기 때문에 따로 전에 초기화할 필요가 없음
				, hidden_to_gate.GetBuffer(), m_total_gate_size)
				))
			{
				DEBUG_OUTPUT(L"failed cublasSgemm to calculation of prev hidden for gates. %s", CudaPlatform::GetErrorString().c_str());
				return false;
			}
			ClipAdd << <CudaPlatform::GetCudaBlockCount(m_batch_total_gate_per_time), CudaPlatform::threadsPerBlock >> >
				(m_batch_total_gate_per_time, m_total_gate_size, cont_t, t > 0, hidden_to_gate.GetBuffer(), gate_t);
			if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
			{
				DEBUG_OUTPUT(L"failed apply previous hidden. %s", CudaPlatform::GetErrorString().c_str());;
				return false;
			}
		}

		ActivationForward << <CudaPlatform::GetCudaBlockCount(m_batch_total_gate_per_time), CudaPlatform::threadsPerBlock >> >
			(m_batch_total_gate_per_time, m_hidden_size, gate_t);
		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed calc gate. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}

		LSTMForward << <CudaPlatform::GetCudaBlockCount(m_batch_hidden_per_time), CudaPlatform::threadsPerBlock >> >
			(m_batch_hidden_per_time, m_hidden_size, cont_t, t > 0, prev_c_t, gate_t, c_t, h_t);
		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed calc cell and hidden values. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}

		prev_c_t = c_t;
		prev_h_t = h_t;

		c_t += m_batch_hidden_per_time;
		h_t += m_batch_hidden_per_time;
		gate_t += m_batch_total_gate_per_time;

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

__global__ void LSTMBackward(const int n, const int hidden_size
	, const bool* cont, const bool isNotFirst
	, const neuron_value* c_prev, neuron_value* dc_prev
	, const neuron_value* gate, const neuron_value* c_t
	, neuron_value* dc_t, const neuron_value* dh_t
	, neuron_value* gate_diff) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		const int sample = index / hidden_size;
		const int hidden_index = index % hidden_size;
		const neuron_value* gate_t = gate + 4 * hidden_size*sample;
		const neuron_value tanh_c = tanh(c_t[index]);
		const bool cont_t = cont ? cont[sample] : isNotFirst;
		neuron_value* gate_diff_t = gate_diff + 4 * hidden_size*sample;

		// Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
		dc_t[index] += dh_t[index] * gate_t[2 * hidden_size + hidden_index] * (neuron_value(1) - tanh_c * tanh_c);
		// c_diff(t-1) += f(t) * c_diff(t)
		if (dc_prev)
			dc_prev[index] = cont_t ? dc_t[index] * gate_t[hidden_size + hidden_index] : 0;

		// Input gate : g(t) * c_diff(t)
		gate_diff_t[hidden_index] = dc_t[index] * gate_t[3 * hidden_size + hidden_index];
		// Forget gate : c(t-1) * c_diff(t)
		gate_diff_t[hidden_size + hidden_index] = cont_t && c_prev ? dc_t[index] * c_prev[index] : 0;
		// Output gate : tanh(c(t)) * h_diff(t)
		gate_diff_t[ 2 * hidden_size + hidden_index] = dh_t[index] * tanh_c;
		// Input modulation gate : i(t) * c_diff(t)
		gate_diff_t[3 * hidden_size + hidden_index] = dc_t[index] * gate_t[hidden_index];
	}
}

__global__ void ActivationBackward(const int n, const int hidden_size,
	const neuron_value clip_threshold, const neuron_value* gate, neuron_value* pre_gate_diff)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		const neuron_value gate_val = gate[index];
		if (index % (4 * hidden_size) < 3 * hidden_size)
			pre_gate_diff[index] = pre_gate_diff[index] * gate_val * (neuron_value(1) - gate_val);
		else 
			pre_gate_diff[index] = pre_gate_diff[index] * (neuron_value(1) - gate_val * gate_val);

		if (clip_threshold > neuron_value(0)) 
		{
			if (pre_gate_diff[index] < -clip_threshold) 
				pre_gate_diff[index] = -clip_threshold;
			else if (pre_gate_diff[index] > clip_threshold) 
				pre_gate_diff[index] = clip_threshold;
		}
	}
}

bool LstmLayerCudaEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	cublasHandle_t cublasHandle = m_net_param.cuda_instance->cublas_handle();
	if (cublasHandle == NULL)
	{
		DEBUG_OUTPUT(L"no cublas handle");
		return false;
	}

	const neuro_u32 batch_size = output_data.batch_size;

	// 4개의 gate 전 값 계산
	const neuron_weight* weight = m_inner_data_vector[0].data.buffer;
	const neuron_weight* hidden_weight = m_inner_data_vector[2].data.buffer;

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
		DEBUG_OUTPUT(L"failed memcpy. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	const _NEURO_TENSOR_DATA* prev_time_cell_data = NULL;
	const _NEURO_TENSOR_DATA* prev_time_cell_diff = NULL;
	const _NEURO_TENSOR_DATA* prev_time_hidden_diff = NULL;

	if (m_prev_rnn)	// 여기에서 이전 lstm으로 error를 전달해주기 때문에 다음 lstm의 error를 가져올 필요가 없다.
	{
		prev_time_cell_data = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastCellData();
		prev_time_cell_diff = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastCellDiff();
		prev_time_hidden_diff = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastHiddenDiff();
	}
	// 요건 앞뒤 연결이 없을때 하는 건데... 좀더 고민해봐야 함!
	else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)	
	{
		prev_time_cell_data = &m_last_time_cell;
		prev_time_cell_diff = &m_last_time_cell_diff;
		prev_time_hidden_diff = &m_last_time_hidden_diff;
	}

	_VALUE_VECTOR hidden_to_hidden(m_net_param.run_pdtype, true);
	if (!hidden_to_hidden.Alloc(m_batch_hidden_per_time))
	{
		DEBUG_OUTPUT(L"failed alloc hidden_to_hidden buffer(%u)", m_batch_hidden_per_time*sizeof(neuron_value));
		return false;
	}

	bool* cont_t = m_cont_idc ? m_cont_idc->GetBuffer() + (m_time_length - 1) * batch_size : NULL;

	for (int t = m_time_length - 1; t >= 0; --t)
	{
		const neuron_value* c_prev = t > 0 ? c_t - m_batch_hidden_per_time : (prev_time_cell_data ? prev_time_cell_data->GetBuffer() : NULL);

		neuron_value* dh_prev = t > 0 ? dh_t - m_batch_hidden_per_time : (prev_time_hidden_diff ? prev_time_hidden_diff->GetBuffer() : NULL);
		neuron_value* dc_prev = t > 0 ? GetCellDiff(t-1).GetBuffer() : (prev_time_cell_diff ? prev_time_cell_diff->GetBuffer() : NULL);

		neuron_value* dc_t = GetCellDiff(t).GetBuffer();

		LSTMBackward << <CudaPlatform::GetCudaBlockCount(m_batch_hidden_per_time), CudaPlatform::threadsPerBlock >> >
			(m_batch_hidden_per_time, m_hidden_size, cont_t, t>0, c_prev, dc_prev, gate_t, c_t, dc_t, dh_t, pre_gate_diff_t);
		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed LSTMBackward. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}

		ActivationBackward << <CudaPlatform::GetCudaBlockCount(m_batch_total_gate_per_time), CudaPlatform::threadsPerBlock >> >
			(m_batch_total_gate_per_time, m_hidden_size, m_clipping_threshold, gate_t, pre_gate_diff_t);
		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed ActivationBackward. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}

		// Backprop errors to the previous time step
		if (dh_prev)
		{
			if (!CudaPlatform::CublasErrorCheck(
				cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N
				, m_hidden_size
				, batch_size
				, m_total_gate_size
				, &dataType::oneval
				, hidden_weight, m_hidden_size
				, pre_gate_diff_t, m_total_gate_size
				, &dataType::zeroval
				, hidden_to_hidden.GetBuffer(), m_hidden_size)))
			{
				DEBUG_OUTPUT(L"failed backpropagation errors to previous hidden. %s", CudaPlatform::GetErrorString().c_str());;
				return false;
			}

			ClipAdd << <CudaPlatform::GetCudaBlockCount(m_batch_hidden_per_time), CudaPlatform::threadsPerBlock >> >
				(m_batch_hidden_per_time, m_hidden_size, cont_t, t>0, hidden_to_hidden.GetBuffer(), dh_prev);
		}

		c_t -= m_batch_hidden_per_time;
		gate_t -= m_batch_total_gate_per_time;
		dh_t -= m_batch_hidden_per_time;
		pre_gate_diff_t -= m_batch_total_gate_per_time;
		if (cont_t)
			cont_t -= batch_size;
	}

	if (!CudaPlatform::CublasErrorCheck(
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N
		, input_error.value_size
		, m_time_length * batch_size
		, m_total_gate_size
		, &dataType::oneval
		, weight, input_error.value_size
		, m_prev_gate_diff.GetBuffer(), m_total_gate_size
		, &dataType::zeroval
		, input_error.GetBuffer(), input_error.value_size)))
	{
		DEBUG_OUTPUT(L"failed backpropagation errors to input. %s", CudaPlatform::GetErrorString().c_str());;
		return false;
	}

	return true;
}

bool LstmLayerCudaEngine::BackwardWeight(neuro_u32 index
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& input_data
	, const _VALUE_VECTOR& grad_weight)
{
	cublasHandle_t cublasHandle = m_net_param.cuda_instance->cublas_handle();
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
		if (!CudaPlatform::CublasErrorCheck(
			cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T
				, input_size
				, m_total_gate_size
				, m_time_length * batch_size
				, &dataType::oneval
				, input_data.GetBuffer(), input_size
				, m_prev_gate_diff.GetBuffer(), m_total_gate_size
				, &dataType::zeroval
				, grad_weight.buffer, input_size)))
		{
			DEBUG_OUTPUT(L"failed cublasSgemm to get derivation of hiddden to gate weight. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}
	}
	else if (index == 1)
	{
		if (grad_weight.count != m_total_gate_size)
		{
			DEBUG_OUTPUT(L"grad bias's size is strange");
			return false;
		}

		if (!CudaPlatform::CublasErrorCheck(
			cublasSgemv(cublasHandle, CUBLAS_OP_N
				, m_total_gate_size
				, m_time_length * current_error.batch_size
				, &dataType::oneval
				, m_prev_gate_diff.GetBuffer(), m_total_gate_size
				, m_net_param.sdb.one_set_vector.buffer, 1
				, &dataType::zeroval
				, grad_weight.buffer, 1)))
		{
			DEBUG_OUTPUT(L"failed cublasSgemm to get derivation of hiddden to gate bias. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}
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
			prev_time_hidden_data = &((LstmLayerCudaEngine*)m_prev_rnn)->GetLastHiddenData();
		}
		else if (m_cont_idc/* || m_bSequanceInEpoch && !isFirstBatch*/)
		{
			prev_time_hidden_data = &m_last_time_hidden;
		}

		// Gradient w.r.t. hidden-to-hidden weight
		// Add Gradient from previous time-step
		if (!CudaPlatform::CublasErrorCheck(
			cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T
				, m_hidden_size
				, m_total_gate_size
				, (m_time_length - 1)*output_data.batch_size
				, &dataType::oneval
				, output_data.GetBuffer(), m_hidden_size
				, m_prev_gate_diff.GetBuffer() + m_batch_total_gate_per_time, m_total_gate_size
				, &dataType::zeroval
				, grad_weight.buffer, m_hidden_size)))
		{
			DEBUG_OUTPUT(L"failed cublasSgemm to get derivation of hiddden to hidden weight from second time. %s", CudaPlatform::GetErrorString().c_str());;
			return false;
		}
		if (prev_time_hidden_data)
		{
			if (!CudaPlatform::CublasErrorCheck(
				cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T
					, m_hidden_size
					, m_total_gate_size
					, 1
					, &dataType::oneval
					, prev_time_hidden_data->GetBuffer(), m_hidden_size
					, m_prev_gate_diff.GetBuffer(), m_total_gate_size
					, &dataType::oneval
					, grad_weight.buffer, m_hidden_size)))
			{
				DEBUG_OUTPUT(L"failed cublasSgemm to get derivation of hiddden to hidden weight in first time. %s", CudaPlatform::GetErrorString().c_str());;
				return false;
			}
		}
	}
	return true;
}
