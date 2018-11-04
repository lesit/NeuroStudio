#include "stdafx.h"

#include "PoolingLayerCpuEngine.h"

#include "util/cpu_parallel_for.h"

using namespace np::engine;
using namespace np::engine::layers;

PoolingLayerCpuEngine::PoolingLayerCpuEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: PoolingLayerEngine(net_param, layer), m_max_index_vector(net_param.run_pdtype, true)
{
}

PoolingLayerCpuEngine::~PoolingLayerCpuEngine()
{
}

bool PoolingLayerCpuEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	if (m_entry.pooling.type == (neuro_u8) network::_pooling_type::max_pooling)
		m_max_index_vector.Alloc(buf.GetBatchTimeSize(), buf.value_size);

	return true;
}

bool PoolingLayerCpuEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	const neuro_u32 batch_size = input_data.GetBatchTimeSize();

	if (batch_size != output_data.GetBatchTimeSize())
		return false;

	const neuro_u32 in_height = m_in_ts.GetHeight();
	const neuro_u32 in_width = m_in_ts.GetWidth();

	const neuro_u32 channel_count = m_out_ts.GetChannelCount();
	const neuro_u32 out_height = m_out_ts.GetHeight();
	const neuro_u32 out_width = m_out_ts.GetWidth();

	const neuro_32 kernel_height = m_entry.pooling.filter.kernel_height;
	const neuro_32 kernel_width = m_entry.pooling.filter.kernel_width;
	const neuro_32 stride_height = m_entry.pooling.filter.stride_height;
	const neuro_32 stride_width = m_entry.pooling.filter.stride_width;

	const neuro_32 pad_height = 0;
	const neuro_32 pad_width = 0;

	for_i(batch_size, [&](neuro_u32 sample)
	{
		const neuron_value* in_ptr = input_data.GetBatchTimeData(sample);
		// 이상하다!!
		if (in_ptr == NULL)
			return;

		neuron_value* out_ptr = output_data.GetBatchTimeData(sample);

		if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::max_pooling)
		{
			neuro_u32* mask = m_max_index_vector.GetBatchTimeData(sample);
			memset(mask, -1, sizeof(neuro_u32)*m_max_index_vector.value_size);
			output_data.data.mm.DataSet(out_ptr, -FLT_MAX, m_max_index_vector.value_size);
			for (neuro_32 c = 0; c < channel_count; ++c)
			{
				for (neuro_32 ph = 0; ph < out_height; ++ph)
				{
					for (neuro_32 pw = 0; pw < out_width; ++pw)
					{
						neuro_32 hstart = ph * stride_height - pad_height;
						neuro_32 wstart = pw * stride_width - pad_width;
						neuro_32 hend = min(hstart + kernel_height, in_height);
						neuro_32 wend = min(wstart + kernel_width, in_width);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);

						const neuro_32 pool_index = ph * out_width + pw;
						for (neuro_32 h = hstart; h < hend; ++h) 
						{
							for (neuro_32 w = wstart; w < wend; ++w)
							{
								const neuro_32 in_index = h * in_width + w;
								if (in_ptr[in_index] > out_ptr[pool_index])
								{
									out_ptr[pool_index] = in_ptr[in_index];
									mask[pool_index] = in_index;
								}
							}
						}
					}
				}
				// compute offset
				in_ptr += in_height*in_width;
				out_ptr += out_height*out_width;

				mask += out_height*out_width;
			}
		}
		else if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::ave_pooling)
		{
			for (neuro_32 c = 0; c < channel_count; ++c) 
			{
				for (neuro_32 ph = 0; ph < out_height; ++ph) 
				{
					for (neuro_32 pw = 0; pw < out_width; ++pw) 
					{
						neuro_32 hstart = ph * stride_height - pad_height;
						neuro_32 wstart = pw * stride_width - pad_width;
						neuro_32 hend = min(hstart + kernel_height, in_height + pad_height);
						neuro_32 wend = min(wstart + kernel_width, in_width + pad_width);
						neuro_32 pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, in_height);
						wend = min(wend, in_width);
						for (neuro_32 h = hstart; h < hend; ++h) 
						{
							for (neuro_32 w = wstart; w < wend; ++w) 
							{
								out_ptr[ph * out_width + pw] +=	in_ptr[h * in_width + w];
							}
						}
						out_ptr[ph * out_width + pw] /= pool_size;
					}
				}
				// compute offset
				in_ptr += in_height*in_width;
				out_ptr += out_height*out_width;
			}
		}
	});

	return true;
}

bool PoolingLayerCpuEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	const neuro_u32 batch_size = current_error.GetBatchTimeSize();

	const neuro_u32 in_height = m_in_ts.GetHeight();
	const neuro_u32 in_width = m_in_ts.GetWidth();

	const neuro_u32 channel_count = m_out_ts.GetChannelCount();
	const neuro_u32 out_height = m_out_ts.GetHeight();
	const neuro_u32 out_width = m_out_ts.GetWidth();

	const neuro_32 kernel_height = m_entry.pooling.filter.kernel_height;
	const neuro_32 kernel_width = m_entry.pooling.filter.kernel_width;
	const neuro_32 stride_height = m_entry.pooling.filter.stride_height;
	const neuro_32 stride_width = m_entry.pooling.filter.stride_width;

	const neuro_32 pad_height = 0;
	const neuro_32 pad_width = 0;

	bool bRet = true;
	for_i(batch_size, [&](neuro_u32 sample)
	{
		const neuron_value* delta = current_error.GetBatchTimeData(sample);

		neuron_value* in_error = input_error.GetBatchTimeData(sample);
		if (!in_error)
		{
			DEBUG_OUTPUT(L"no input error buffer pointer");

			bRet = false;
			return;
		}

		if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::max_pooling)
		{
			neuro_u32* mask = m_max_index_vector.GetBatchTimeData(sample);

			for (neuro_32 c = 0; c < channel_count; ++c)
			{
				for (neuro_32 ph = 0; ph < out_height; ++ph) 
				{
					for (neuro_32 pw = 0; pw < out_width; ++pw) 
					{
						const neuro_32 index = ph * out_width + pw;
						const neuro_32 bottom_index = mask[index];
						in_error[bottom_index] += delta[index];
					}
				}
				in_error += in_height*in_width;
				delta += out_height*out_width;

				mask += out_height*out_width;
			}
		}
		else if (m_entry.pooling.type == (neuro_u8)network::_pooling_type::ave_pooling)
		{
			for (neuro_32 c = 0; c < channel_count; ++c) 
			{
				for (neuro_32 ph = 0; ph < out_height; ++ph) 
				{
					for (neuro_32 pw = 0; pw < out_width; ++pw) 
					{
						neuro_32 hstart = ph * stride_height - pad_height;
						neuro_32 wstart = pw * stride_width - pad_width;
						neuro_32 hend = min(hstart + kernel_height, in_height + pad_height);
						neuro_32 wend = min(wstart + kernel_width, in_width + pad_width);
						neuro_32 pool_size = (hend - hstart) * (wend - wstart);
						hstart = max(hstart, 0);
						wstart = max(wstart, 0);
						hend = min(hend, in_height);
						wend = min(wend, in_width);
						for (neuro_32 h = hstart; h < hend; ++h) 
						{
							for (neuro_32 w = wstart; w < wend; ++w) 
								in_error[h * in_width + w] += delta[ph * out_width + pw] / pool_size;
						}
					}
				}
				// offset
				in_error += in_height*in_width;
				delta += out_height*out_width;
			}
		}
	});

	return bRet;
}
