#include "stdafx.h"
#include "cpu_activations.h"

#include "util/cpu_parallel_for.h"

namespace np
{
	namespace engine
	{
		namespace activation
		{
			ActivationFunction* ActivationFunction::CreateInstanceCPU(network::_activation_type type)
			{
				switch (type)
				{
				case network::_activation_type::sigmoid:
					return new cpu::SigmoidActivation;
				case network::_activation_type::reLU:
					return new cpu::ReLuActivation;
				case network::_activation_type::leakyReLU:
					return new cpu::LeakyReLuActivation;
				case network::_activation_type::eLU:
					return new cpu::ELuActivation;
				case network::_activation_type::softmax:
					return new cpu::SoftmaxActivation;
				case network::_activation_type::tahn:
					return new cpu::TanhActivation;
				}
				return NULL;
			}
			neuron_value ActivationFunction::m_relu_negative_slope = 0.01f;
			neuron_value ActivationFunction::m_elu_alpha = 1.f;
		}
	}
}

using namespace np::engine::activation;

bool cpu::OneHotActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	for_i(value.GetSize(), [&](neuro_size_t i)
	{
		value.GetBuffer()[i] = ComputeActivation(value.GetBuffer()[i]);
	});

	return true;
}

bool cpu::OneHotActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	for_i(error.GetSize(), [&](neuro_size_t i)
	{
		error.GetBuffer()[i] = error.GetBuffer()[i] * ComputeDerivative(out.GetBuffer()[i]);
	});

#if defined(_DEBUG_TRAIN)
	std::wstring log = L"\r\ncalc back activation :\r\n";
	for (neuro_size_t c = 0; c < count; c++)
		log += util::StringUtil::Format<wchar_t>(L"error:%+.8f * derivate(%+.8f):%+.8f = %+.8f\r\n", error[c], out[c], ComputeDerivative(out[c]), delta[c]);

	log += L"\r\n";
	DEBUG_OUTPUT(log.c_str());
#endif
	return true;
}

inline neuron_value cpu::SigmoidActivation::ComputeActivation(const neuron_value& value)
{
	return neuron_value(1) / (neuron_value(1) + std::exp(-value));
}

inline neuron_value cpu::SigmoidActivation::ComputeDerivative(const neuron_value& value)
{
	return value * (neuron_value(1) - value);
}

inline neuron_value cpu::TanhActivation::ComputeActivation(const neuron_value& value)
{
	return tanh(value);
}

inline neuron_value cpu::TanhActivation::ComputeDerivative(const neuron_value& value)
{
	return neuron_value(1) - value * value;
}

inline neuron_value cpu::ReLuActivation::ComputeActivation(const neuron_value& value)
{
	return value>neuron_value(0) ? value : neuron_value(0);
}

inline neuron_value cpu::ReLuActivation::ComputeDerivative(const neuron_value& value)
{
	return value>neuron_value(0) ? neuron_value(1) : neuron_value(0);
}

inline neuron_value cpu::LeakyReLuActivation::ComputeActivation(const neuron_value& value)
{
	return (value > neuron_value(0)) ? value : m_relu_negative_slope * value;
}

inline neuron_value cpu::LeakyReLuActivation::ComputeDerivative(const neuron_value& value)
{
	return (value > neuron_value(0)) ? neuron_value(1) : m_relu_negative_slope;	
}

inline neuron_value cpu::ELuActivation::ComputeActivation(const neuron_value& value)
{
	return (value > neuron_value(0) ? value : m_elu_alpha * (exp(value) - neuron_value(1)));
}

inline neuron_value cpu::ELuActivation::ComputeDerivative(const neuron_value& value)
{
	return (value > neuron_value(0) ? neuron_value(1) : (m_elu_alpha + value));
}

cpu::SoftmaxActivation::SoftmaxActivation()
: m_add_buf(core::math_device_type::cpu, true)
{
}

cpu::SoftmaxActivation::~SoftmaxActivation()
{

}

bool cpu::SoftmaxActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	for_i(value.GetBatchSize(), [&](neuro_u32 sample)
	{
		neuron_value* start = value.GetBatchData(sample);
		neuron_value* end = start + value.value_size;

		// prevent explording
		neuron_value max = *std::max_element(start, end);

		neuron_value sum = 0;

		for (neuron_value* ptr = start; ptr != end; ptr++)
		{
			*ptr = exp(*ptr - max);

			sum += *ptr;
		}

		for (neuron_value* ptr = start; ptr != end; ptr++)
			*ptr /= sum;
	});

	return true;
}

#include "core/MathCpuCore.h"

bool cpu::SoftmaxActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	np::core::MathCpuCore cpu_math;
	for_i(error.GetBatchSize(), [&](neuro_u32 sample)
	{
		const neuron_value* s_out = out.GetBatchData(sample);
		neuron_value* s_error = error.GetBatchData(sample);

		neuron_value scale;
		cpu_math.dot(error.value_size, s_error, s_out, scale);

		for (neuro_size_t index = 0; index < error.value_size; index++)
		{
			s_error[index] = (s_error[index] - scale)*s_out[index];
		}
	});

	return true;
}
