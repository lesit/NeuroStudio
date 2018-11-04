#include "stdafx.h"

#include "BatchNormLayerEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

BatchNormLayerEngine* BatchNormLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	return new BatchNormLayerEngine(net_param, layer);
}

BatchNormLayerEngine::BatchNormLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
:HiddenLayerEngine(net_param, layer)
, m_mean(net_param.run_pdtype, true)
, m_variance(net_param.run_pdtype, true)
, m_temp(net_param.run_pdtype, true)
, m_cpu_mean_div_variance(core::math_device_type::cpu, true)
{
	m_batch_size = 0;
	m_channel = 0;
	m_spatial_dim = 0;
}

BatchNormLayerEngine::~BatchNormLayerEngine()
{
}

bool BatchNormLayerEngine::OnInitialized()
{
	if (!__super::OnInitialized())
		return false;

	if (m_in_ts.GetHeight() > 1 && m_in_ts.GetWidth() > 1)	// 3차원 이상으로 실제 channel이 있을 경우
	{
		m_channel = m_in_ts.GetChannelCount();
		m_spatial_dim = m_in_ts.GetHeight()*m_in_ts.GetWidth();
	}
	else // 1차원 일 경우
	{
		m_channel = 1;
		m_spatial_dim = m_in_ts.GetDimSize();
	}

	m_mean.Alloc(m_channel);
	m_variance.Alloc(m_channel);

	m_cpu_mean_div_variance.Alloc(1);
	return true;
}

neuro_u32 BatchNormLayerEngine::Get1MultiplierSize() const
{
	return m_spatial_dim;
}

neuro_u32 BatchNormLayerEngine::Get1MultiplierSizePerBatch() const
{
	return max(m_in_ts.time_length, 1);
}

bool BatchNormLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	m_batch_size = buf.GetBatchTimeSize();

	if (!m_temp.AllocLike(buf))
	{
		DEBUG_OUTPUT(L"failed alloc temporary memry");
		return false;
	}

	return true;
}

#define _MEAN_VAR_DIVIDE_VERSION

/*
	Basically, Batch Normalization comes from Caffe.
	Actually, I wanted to use it without saving the mean div variance part, but I had a problem that did not work well.
	So I will define _MEAN_VAR_DIVIDE_VERSION and save the original Caffe source and fix it later through research.
*/
bool BatchNormLayerEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	const _VALUE_VECTOR& save_mean = m_inner_data_vector[0].data;
	const _VALUE_VECTOR& save_variance = m_inner_data_vector[1].data;

#ifdef _MEAN_VAR_DIVIDE_VERSION
	m_cpu_mean_div_variance.CopyFrom(m_inner_data_vector[2].data);
#endif

	if (input_data.GetBuffer() != output_data.GetBuffer())
		output_data.CopyFrom(input_data);

	_VALUE_VECTOR num_by_chans(m_net_param.run_pdtype, true);
	num_by_chans.Alloc(m_batch_size * m_channel);
	if (bTrain)
	{
		// compute mean
		m_net_param.math.gemv(CblasNoTrans, m_batch_size * m_channel, m_spatial_dim
			, 1.f / (m_batch_size * m_spatial_dim), input_data.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer()
			, 0.f, num_by_chans.GetBuffer());
		m_net_param.math.gemv(CblasTrans, m_batch_size, m_channel
			, 1.f, num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer()
			, 0.f, m_mean.GetBuffer());
	}
	else
	{
#if 0//defined(_DEBUG)
		static BatchNormLayerEngine* firstBN = NULL;
		if (!firstBN)
			firstBN = this;
		
		if (firstBN == this)
		{
			static neuro_size_t i_test = 1;
			if (i_test % 4 == 0)
			{
				_VALUE_VECTOR temp_save_mean(core::math_device_type::cpu, true); temp_save_mean.Alloc(save_mean.count); temp_save_mean.CopyFrom(save_mean);
				_VALUE_VECTOR temp_save_variance(core::math_device_type::cpu, true); temp_save_variance.Alloc(save_variance.count); temp_save_variance.CopyFrom(save_variance);

				DEBUG_OUTPUT(L"saved mean :\t %f", *temp_save_mean.buffer);
				DEBUG_OUTPUT(L"saved variance :\t %f", *temp_save_variance.buffer);
			}
			++i_test;
		}
#endif

		m_mean.CopyFrom(save_mean);
		m_variance.CopyFrom(save_variance);

#ifdef _MEAN_VAR_DIVIDE_VERSION
		neuro_float scale_factor = *m_cpu_mean_div_variance.buffer == 0 ? 0 : 1 / *m_cpu_mean_div_variance.buffer;
		m_net_param.math.scale(m_mean.count, scale_factor, m_mean.buffer);
		m_net_param.math.scale(m_variance.count, scale_factor, m_variance.buffer);
#endif
	}

	// subtract mean
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_batch_size, m_channel, 1
		, 1.f, m_net_param.sdb.one_set_vector.GetBuffer(), m_mean.GetBuffer()
		, 0.f, num_by_chans.GetBuffer());
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_channel * m_batch_size, m_spatial_dim, 1
		, -1.f, num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer()
		, 1.f, output_data.GetBuffer());

	if (bTrain)
	{
		// compute variance using var(X) = E((X-EX)^2)
		m_net_param.math.powx(output_data.GetSize(), output_data.GetBuffer(), neuro_float(2),
			m_temp.GetBuffer());  // (X-EX)^2
		m_net_param.math.gemv(CblasNoTrans, m_channel * m_batch_size, m_spatial_dim,
			1.f / (m_batch_size * m_spatial_dim), m_temp.GetBuffer(),
			m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
			num_by_chans.GetBuffer());
		m_net_param.math.gemv(CblasTrans, m_batch_size, m_channel, 1.f,
			num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
			m_variance.GetBuffer());  // E((X_EX)^2)

		// compute and save moving average
		neuro_u32 m = input_data.GetSize() / m_channel;
		neuro_float bias_correction_factor = m > 1 ? neuro_float(m) / (m - 1) : 1;

#ifdef _MEAN_VAR_DIVIDE_VERSION
		*m_cpu_mean_div_variance.buffer *= m_entry.batch_norm.momentum;
		*m_cpu_mean_div_variance.buffer += 1;
		m_inner_data_vector[2].data.CopyFrom(m_cpu_mean_div_variance);
		m_net_param.math.axpby(m_mean.count, neuro_float(1), m_mean.buffer, m_entry.batch_norm.momentum, save_mean.buffer);
		m_net_param.math.axpby(m_variance.count, bias_correction_factor, m_variance.buffer, m_entry.batch_norm.momentum, save_variance.buffer);
#else
		bias_correction_factor *= (neuro_float(1) - m_entry.batch_norm.momentum);
		m_net_param.math.axpby(m_mean.count, neuro_float(1) - m_entry.batch_norm.momentum, m_mean.buffer, m_entry.batch_norm.momentum, save_mean.buffer);
		m_net_param.math.axpby(m_variance.count, bias_correction_factor, m_variance.buffer, m_entry.batch_norm.momentum, save_variance.buffer);
#endif
	}

	// normalize variance
	m_net_param.math.add_scalar(m_variance.count, m_entry.batch_norm.eps, m_variance.GetBuffer());
	m_net_param.math.powx(m_variance.count, m_variance.GetBuffer(), neuro_float(0.5),
		m_variance.GetBuffer());

	// replicate variance to input size
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_batch_size, m_channel, 1
		, 1.f, m_net_param.sdb.one_set_vector.GetBuffer(), m_variance.GetBuffer()
		, 0.f, num_by_chans.GetBuffer());
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_channel * m_batch_size, m_spatial_dim, 1
		, 1.f, num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer()
		, 0.f, m_temp.GetBuffer());
	m_net_param.math.div(output_data.GetSize(), output_data.GetBuffer(), m_temp.GetBuffer(), output_data.GetBuffer());
	// TODO(cdoersch): The caching is only needed because later in-place layers
	//                 might clobber the data.  Can we skip this if they won't?

	return true;
}

bool BatchNormLayerEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	_VALUE_VECTOR num_by_chans(m_net_param.run_pdtype, true);
	num_by_chans.Alloc(m_batch_size * m_channel);

	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	//
	// dE(Y)/dX =
	//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
	//     ./ sqrt(var(X) + eps)
	//
	// where \cdot and ./ are hadamard product and elementwise division,
	// respectively, dE/dY is the top diff, and mean/var/sum are all computed
	// along all dimensions except the channels dimension.  In the above
	// equation, the operations allow for expansion (i.e. broadcast) along all
	// dimensions except the channels dimension where required.

	// sum(dE/dY \cdot Y)
	m_net_param.math.mul(current_error.GetSize(), output_data.GetBuffer(), current_error.GetBuffer(), input_error.GetBuffer());
	m_net_param.math.gemv(CblasNoTrans, m_channel * m_batch_size, m_spatial_dim, 1.f,
		input_error.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
		num_by_chans.GetBuffer());
	m_net_param.math.gemv(CblasTrans, m_batch_size, m_channel, 1.f,
		num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
		m_mean.GetBuffer());

	// reshape (broadcast) the above
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_batch_size, m_channel, 1, 1,
		m_net_param.sdb.one_set_vector.GetBuffer(), m_mean.GetBuffer(), 0.f,
		num_by_chans.GetBuffer());
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_channel * m_batch_size,
		m_spatial_dim, 1, 1.f, num_by_chans.GetBuffer(),
		m_net_param.sdb.one_set_vector.GetBuffer(), 0.f, input_error.GetBuffer());

	// sum(dE/dY \cdot Y) \cdot Y
	m_net_param.math.mul(current_error.GetSize(), output_data.GetBuffer(), input_error.GetBuffer(), input_error.GetBuffer());

	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	m_net_param.math.gemv(CblasNoTrans, m_channel * m_batch_size, m_spatial_dim, 1.f,
		current_error.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
		num_by_chans.GetBuffer());
	m_net_param.math.gemv(CblasTrans, m_batch_size, m_channel, 1.f,
		num_by_chans.GetBuffer(), m_net_param.sdb.one_set_vector.GetBuffer(), 0.f,
		m_mean.GetBuffer());
	// reshape (broadcast) the above to make
	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_batch_size, m_channel, 1, 1,
		m_net_param.sdb.one_set_vector.GetBuffer(), m_mean.GetBuffer(), 0.f,
		num_by_chans.GetBuffer());
	m_net_param.math.gemm(CblasNoTrans, CblasNoTrans, m_batch_size * m_channel,
		m_spatial_dim, 1, 1.f, num_by_chans.GetBuffer(),
		m_net_param.sdb.one_set_vector.GetBuffer(), 1.f, input_error.GetBuffer());

	// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
	m_net_param.math.axpby(current_error.GetSize(), neuro_float(1), current_error.GetBuffer(),
		neuro_float(-1.f / (m_batch_size * m_spatial_dim)), input_error.GetBuffer());

	// note: m_temp still contains sqrt(var(X)+eps), computed during the forward
	// pass.
	m_net_param.math.div(current_error.GetSize(), input_error.GetBuffer(), m_temp.GetBuffer(), input_error.GetBuffer());

	return true;
}
