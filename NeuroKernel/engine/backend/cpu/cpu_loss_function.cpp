#include "stdafx.h"

#include "cpu_loss_function.h"
#include "util/cpu_parallel_for.h"

using namespace np::engine;
using namespace np::engine::loss;
using namespace np::engine::loss::cpu;

using namespace std;

LossFunction* LossFunction::CreateInstanceCPU(network::_loss_type type, bool read_label_for_target)
{
	switch (type)
	{
	case network::_loss_type::CrossEntropy:
		return new cpu::CrossEntropy(read_label_for_target);
	case network::_loss_type::CrossEntropyMulticlass:
		return new cpu::CrossEntropyMulticlass(read_label_for_target);
	default:	// case network::_loss_type::MSE:
		return new cpu::MSE(read_label_for_target);
	}
}

neuron_error MSE::CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target)
{
	neuron_error d = 0.0;

	if (m_read_label_for_target)
	{
		for (neuro_u32 sample = 0; sample < batch_size; sample++)
		{
			const neuro_u32 label = ((const neuro_u32*)target)[sample];
			for (neuro_u32 i = 0, n = value_size; i < n; ++i, output++)
			{
				neuro_float target_value = label == i ? 1.f : 0.f;
				d += (*output - target_value) * (*output - target_value);
			}
		}
	}
	else
	{
		const neuro_float* target_ptr = (const neuro_float*)target;
		for (neuro_u32 i = 0, n = batch_size*value_size; i < n; ++i)
			d += (output[i] - target_ptr[i]) * (output[i] - target_ptr[i]);
	}

	return d / 2;
}

bool MSE::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	const neuron_error scale = normalize_factor(batch_size);

#if 0//defined(_DEBUG)
	if (m_read_label_for_target)
	{
		const neuro_u32* label_vector = (const neuro_u32*)target;
		for (neuro_size_t index = 0, n = batch_size*value_size; index < n; index++)
		{
			const neuro_size_t sample = index / value_size;
			const neuro_size_t sample_index = index % value_size;
			neuro_float target_value = label_vector[sample] == sample_index ? 1.f : 0.f;

			diff[index] = scale * (output[index] - target_value);
		}
		DEBUG_OUTPUT(L"");
		NP_Util::DebugOutputValues(label_vector, batch_size, 10);
		DEBUG_OUTPUT(L"");
		NP_Util::DebugOutputValues(output, batch_size*value_size, 10);
		DEBUG_OUTPUT(L"");
		NP_Util::DebugOutputValues(diff, batch_size*value_size, 10);
	}
#endif

	for_i(batch_size, [&](neuro_u32 sample)
	{
		const neuro_float* output_ptr = output + sample * value_size;
		neuron_error* diff_ptr = diff + sample * value_size;
		if (m_read_label_for_target)
		{
			const neuro_u32 label = ((const neuro_u32*)target)[sample];
			for (neuro_size_t i = 0; i < value_size; ++i)
				diff_ptr[i] = scale * (output_ptr[i] - (label == i ? 1.f : 0.f));
		}
		else
		{
			const neuro_float* target_ptr = (const neuro_float*)target + sample * value_size;
			for (neuro_size_t i = 0; i < value_size; ++i)
				diff_ptr[i] = scale * (output_ptr[i] - target_ptr[i]);
		}
	});

	return true;
}

// cross-entropy loss function for (multiple independent) binary classifications. sigmoid¿¡ »ç¿ëµÊ
neuron_error CrossEntropy::CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target)
{
	neuron_error loss = 0.f;

	if (m_read_label_for_target)
	{
		for (neuro_u32 sample = 0; sample < batch_size; sample++)
		{
			const neuro_u32 label = ((const neuro_u32*)target)[sample];

			for (neuro_u32 i = 0, n = value_size; i < n; ++i, output++)
			{
				neuro_float target_value = label == i ? 1.f : 0.f;
				loss -= target_value * log(max(*output, FLT_MIN)) + (1.f - target_value) * log(max(1.f - *output, FLT_MIN));
			}
		}
	}
	else
	{
		const neuro_float* target_ptr = (const neuro_float*)target;

		for (neuro_u32 i = 0, n = batch_size*value_size; i < n; ++i)
			loss -= target_ptr[i] * log(max(output[i], FLT_MIN)) + (1.f - target_ptr[i]) * log(max(1.f - output[i], FLT_MIN));
	}

	return loss;
}

bool CrossEntropy::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	const neuron_error scale = normalize_factor(batch_size);

	for_i(batch_size, [&](neuro_u32 sample)
	{
		const neuro_float* output_ptr = output + sample * value_size;
		neuron_error* diff_ptr = diff + sample * value_size;
		if (m_read_label_for_target)
		{
			neuro_u32 label = ((const neuro_u32*)target)[sample];
			for (neuro_size_t i = 0; i < value_size; ++i)
				diff_ptr[i] = scale * (label == i ? 1.f : 0.f) / max(output_ptr[i], FLT_MIN);
		}
		else
		{
			const neuro_float* target_ptr = (const neuro_float*)target + sample * value_size;
			for (neuro_size_t i = 0; i < value_size; ++i)
				diff_ptr[i] = scale * target_ptr[i] / max(output_ptr[i], FLT_MIN);
		}
	});

	return true;
}

// cross-entropy loss function for multi-class classification. ¼ÒÇÁÆ®¸Æ½º¿¡ »ç¿ëµÊ
neuron_error CrossEntropyMulticlass::CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target)
{
	neuron_error loss = 0.f;

	if (m_read_label_for_target)
	{
		for (neuro_u32 sample = 0; sample < batch_size; ++sample)
		{
			neuro_u32 label = ((const neuro_u32*)target)[sample];
			loss -= log(max(output[label], FLT_MIN));

			output += value_size;
		}

	}
	else
	{
		const neuro_float* target_ptr = (const neuro_float*)target;

		for (neuro_u32 i = 0; i < batch_size; ++i)
		{
			neuro_u32 label = np::max_index(target_ptr, value_size);
			loss -= log(max(output[label], FLT_MIN));

			output += value_size;
			target_ptr += value_size;
		}
	}
	return loss;
}

bool CrossEntropyMulticlass::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	const neuron_error scale = normalize_factor(batch_size);

	for_i(batch_size, [&](neuro_u32 sample)
	{
		neuro_u32 label = 0;
		if (m_read_label_for_target)
		{
			label = ((const neuro_u32*)target)[sample];
		}
		else
		{
			const neuro_float* target_p = ((const neuro_float*)target) + sample*value_size;
			label = max_index(target_p, value_size);
		}

		neuro_float prob = max(output[sample*value_size + label], FLT_MIN);

		diff[sample*value_size + label] = scale / prob;

	});

	return true;
}
