#include "stdafx.h"
#include "optimizer.h"

using namespace np::engine;
using namespace np::engine::optimizer;
OptimizeInEpoch* OptimizeInEpoch::CreateInstance(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance
	, _optimizer_type type, const std::vector<neuro_float>& parameters, const network::_OPTIMIZING_RULE& rule)
{
	Optimizer* optimizer;
	if (pdtype == core::math_device_type::cuda)
		optimizer = Optimizer::CreateInstanceCUDA(type, parameters);
	else
		optimizer = Optimizer::CreateInstanceCPU(type, parameters);

	if (optimizer == NULL)
		return NULL;

	return new OptimizeInEpoch(pdtype, cuda_instance, optimizer, rule);
}

OptimizeInEpoch::OptimizeInEpoch(core::math_device_type pdtype, core::cuda::CudaInstance* cuda_instance, Optimizer* optimizer
	, const network::_OPTIMIZING_RULE& _rule)
	: m_math(pdtype, cuda_instance)
{
	m_optimizer = optimizer;

	memcpy(&m_rule, &_rule, sizeof(network::_OPTIMIZING_RULE));

	m_rule.lr_policy.lr_base = m_optimizer->default_learn_rate();
	if (_rule.lr_policy.lr_base > 0.0f)
		m_rule.lr_policy.lr_base = _rule.lr_policy.lr_base;

	if (m_rule.lr_policy.type != network::_lr_policy_type::Fix)
		m_rule.lr_policy.step = 1;

	if (m_rule.wn_policy.weight_decay == 0.f)
		m_rule.wn_policy.type = network::_wn_policy_type::none;

	m_current_iterator = 0;

	m_learn_rate = m_rule.lr_policy.lr_base;
	if (m_learn_rate<minimum_learn_rate)
		m_learn_rate = minimum_learn_rate;

	DEBUG_OUTPUT(L"lr : %f, lr policy type : %u, lr gamma : %f, lr step : %u, weight norm type : %u, weight decay : %f"
		, m_learn_rate, m_rule.lr_policy.type, m_rule.lr_policy.gamma, m_rule.lr_policy.step, m_rule.wn_policy.type, m_rule.wn_policy.weight_decay);
}

OptimizeInEpoch::~OptimizeInEpoch()
{
	delete m_optimizer;
}

#include "util/randoms.h"
void OptimizeInEpoch::NextBatch()
{
	if (m_rule.lr_policy.type == network::_lr_policy_type::Random)
	{
		m_learn_rate = uniform_rand(neuro_float(0.00000001), m_rule.lr_policy.lr_base);
	}
	else if (m_rule.lr_policy.type == network::_lr_policy_type::StepByBatch)
	{
		++m_current_iterator;
		m_learn_rate = m_rule.lr_policy.lr_base * pow(m_rule.lr_policy.gamma, m_current_iterator);
	}
}

void OptimizeInEpoch::NextEpoch()
{
	if (m_rule.lr_policy.type == network::_lr_policy_type::StepByEpoch)
	{
		++m_current_iterator;
		m_learn_rate = m_rule.lr_policy.lr_base * pow(m_rule.lr_policy.gamma, m_current_iterator);
	}

	m_optimizer->NextEpoch();
}

bool OptimizeInEpoch::Update(neuro_float lr_mult, neuron_weight* gradient, const _LAYER_INNER_DATA& weight)
{
	// 이미 loss에서 적용했음. 즉 여기서 이럴 필요 없음!!
//	m_optimizer->m_math.scale(count, norm_factor, gradient);

	if (m_rule.wn_policy.type != network::_wn_policy_type::none && weight.wtype == network::_layer_data_type::weight)
	{
		// 대신 loss 에 1/2 * weight_decay * W^2 을 더해야 한다.
		if (m_rule.wn_policy.type == network::_wn_policy_type::L2)
		{
			if (!m_math.axpy(weight.data.count, m_rule.wn_policy.weight_decay, weight.data.buffer, gradient))
			{
				DEBUG_OUTPUT(L"failed Regularize");
				return false;
			}
		}
	}

	if (!m_optimizer->Update(m_learn_rate*lr_mult, gradient, weight))
	{
		DEBUG_OUTPUT(L"failed update weight");
		return false;
	}
	return true;
}
