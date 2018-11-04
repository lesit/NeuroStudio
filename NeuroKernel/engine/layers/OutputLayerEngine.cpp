#include "stdafx.h"

#include "OutputLayerEngine.h"

#include "core/cuda_platform.h"

#include "../../network/OutputLayer.h"

using namespace np::engine;
using namespace np::engine::layers;

OutputLayerEngine* OutputLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	return new OutputLayerEngine(net_param, layer);
}

OutputLayerEngine::OutputLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
	: HiddenLayerEngine(net_param, layer), m_target_buffer(net_param.run_pdtype, true)
	, m_read_label_for_target(((const network::OutputLayer&)layer).ReadLabelForTarget())
{
	m_loss_function = NULL;
	m_diff_function = NULL;

	m_is_backward_activation = false;

	m_loss = 0;
}

OutputLayerEngine::~OutputLayerEngine()
{
	delete m_loss_function;

	if (m_diff_function != m_loss_function)
		delete m_diff_function;
}

bool OutputLayerEngine::OnInitialized()
{
	m_is_backward_activation = m_activation != NULL;

	_loss_type loss_type = (_loss_type)m_entry.output.loss_type;

	if (loss_type == _loss_type::SigmoidCrossEntropy ||
		loss_type == _loss_type::SoftmaxWithLoss)
	{
		if (loss_type == _loss_type::SigmoidCrossEntropy)
			loss_type = _loss_type::CrossEntropy;
		else if (loss_type == _loss_type::SoftmaxWithLoss)
			loss_type = _loss_type::CrossEntropyMulticlass;

		// 결합된 loss 인경우엔 diff는 MSE 를 따른다.
		m_diff_function = loss::LossFunction::CreateInstance(m_net_param.run_pdtype, m_net_param.cuda_instance, _loss_type::MSE, m_read_label_for_target);
		if (!m_diff_function)
		{
			DEBUG_OUTPUT(L"failed create loss function(MSE) instance.");
			return false;
		}

		m_is_backward_activation = false;	// loss의 backward를 대신한다. 즉, MSE의 diff만 사용한다.
	}

	m_loss_function = loss::LossFunction::CreateInstance(m_net_param.run_pdtype, m_net_param.cuda_instance, loss_type, m_read_label_for_target);
	if (!m_loss_function)
	{
		DEBUG_OUTPUT(L"failed create loss function(%s) instance.", ToString(loss_type));
		return false;
	}

	if (m_diff_function == NULL)
		m_diff_function = m_loss_function;

	return true;
}

bool OutputLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	if (m_read_label_for_target)
		return m_target_buffer.Calloc(buf.batch_size, buf.time_length, 1) != NULL;
	else
		return m_target_buffer.Alloc(buf.batch_size, buf.time_length, buf.value_size) != NULL;
}

bool OutputLayerEngine::Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output)
{
	_NEURO_TENSOR_DATA input;
	if (!GetInputData(m_input_vector[0], input))
	{
		DEBUG_OUTPUT(L"no input data");
		return false;
	}

	if (!output.CopyFrom(input))
	{
		return false;
	}

#if 0//defined(_DEBUG)
	_NEURO_TENSOR_DATA temp;
	temp.AllocLike(output);
	temp.CopyFrom(output);
	DEBUG_OUTPUT(L"output1");
	NP_Util::DebugOutputValues(temp.GetBuffer(), temp.GetSize(), 10);
#endif
	if (m_activation)
	{
		if (!m_activation->ForwardActivations(output))
		{
			DEBUG_OUTPUT(L"%s, failed ForwardActivations : %s", GetLayerName(), core::cuda::CudaPlatform::GetErrorString().c_str());
			return false;
		}
	}
#if 0//defined(_DEBUG)
	temp.CopyFrom(output);
	DEBUG_OUTPUT(L"output2");
	NP_Util::DebugOutputValues(temp.GetBuffer(), temp.GetSize(), 10);
	temp.Dealloc();
#endif

	if (m_loss_function)
		m_loss = m_loss_function->CalcLoss(batch_size, output.value_size, output.GetBuffer(), m_target_buffer.GetBuffer());

	return true;
}

bool OutputLayerEngine::Backward(neuro_u32 batch_size)
{
	_NEURO_TENSOR_DATA output;
	if (!m_net_param.GetDeviceBuffer(GetOutputData(), TensorBatchTimeOrder(), output))
	{
		DEBUG_OUTPUT(L"failed to get data buffer");
		return false;
	}

	const _NEURO_TENSOR_DATA& input_error = ((HiddenLayerEngine*)m_input_vector[0].engine)->GetErrorBuffer();
	if (!m_diff_function->CalcDiff(batch_size, output.value_size, output.GetBuffer(), m_target_buffer.GetBuffer(), input_error.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed calc output gradient");
		return false;
	}

	if (m_is_backward_activation)
	{
		// activation전의 값인 input으로 해야하나 고민했으나, 이게 맞는것 같다!.
		if (!m_activation->BackwardActivations(output, input_error))
		{
			DEBUG_OUTPUT(L"failed Backward");
			return false;
		}
	}
	return true;
}
