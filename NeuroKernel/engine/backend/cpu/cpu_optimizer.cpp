#include "stdafx.h"

#include "cpu_optimizer.h"

using namespace np::engine;
using namespace np::engine::optimizer;
using namespace np::engine::optimizer::cpu;

Optimizer* Optimizer::CreateInstanceCPU(_optimizer_type type, const std::vector<neuro_float>& parameters)
{
	switch (type)
	{
	case _optimizer_type::Adagrad:
		return new cpu::AdagradCpu;
	case _optimizer_type::Adam:
		return new cpu::AdamCpu(parameters);
	case _optimizer_type::RMSprop:
		return new cpu::RMSpropCpu;
	default:
		return new cpu::SGDCpu;
	}
}

Optimizer::Optimizer()
{

}

bool AdagradCpu::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	/*	ù��° update���� �� �ȵǴ� ������ g�� �ʱⰪ�� 0�̾ dW[2]^2 �غ��� �ʹ� �۾Ƽ�
	�� ������ ������ ������ gradient�� �ʹ� Ŀ�� ������.. �Ѥ�
	���� ������ �ξ��ٰ� ����ϵ��� �Ѵ�.,
	*/
	neuron_weight* g = weight.GetHistory(0);

	for_i(weight.data.count, [&](neuro_u32 i)
	{
		g[i] += dW[i] * dW[i];
		weight.data.buffer[i] -= learn_rate * dW[i] / (std::sqrt(g[i]) + eps);
	});
	return true;
}

bool RMSpropCpu::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* g = weight.GetHistory(0);

	for_i(weight.data.count, [&](neuro_u32 i)
	{
		g[i] = decay * g[i] + (1 - decay) * dW[i] * dW[i];
		weight.data.buffer[i] -= learn_rate * dW[i] / std::sqrt(g[i] + eps);
	});
	return true;
}

bool AdamCpu::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* mt = weight.GetHistory(0);
	neuron_weight* vt = weight.GetHistory(1);

	for_i(weight.data.count, [&](neuro_u32 i){
		mt[i] = beta1 * mt[i] + (neuro_float(1) - beta1) * dW[i];
		vt[i] = beta2 * vt[i] + (neuro_float(1) - beta2) * dW[i] * dW[i];

//		weight.data.buffer[i] -= learn_rate * (mt[i] / (neuro_float(1) - b1_t)) / std::sqrt((vt[i] / (neuro_float(1) - b2_t)) + eps);
		weight.data.buffer[i] -= learn_rate * (mt[i] / (std::sqrt(vt[i]) + eps)) * (std::sqrt(neuro_float(1) - b2_t) / (neuro_float(1) - b1_t));
	});
	return true;
}

bool SGDCpu::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* h = weight.GetHistory(0);

	for_i(weight.data.count, [&](neuro_u32 i)
	{
		h[i] = momentum * h[i] + learn_rate * dW[i];
		weight.data.buffer[i] -= h[i];
	});
	return true;
}
