#include "stdafx.h"

#include "AbstractProducer.h"
#include "NeuroDataFormatProducer.h"
#include "ImageFileProducer.h"
#include "MnistProducer.h"
#include "NlpProducer.h"
#include "NumericProducer.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

AbstractProducer* AbstractProducer::CreateInstance(const char* status
	, DataReaderSet& reader_set
	, const dp::model::AbstractProducerModel& model)
{
	AbstractProducer* producer = NULL;

	JobSignalSender creating_producer_job(reader_set.init_object.GetLongTimeJobSignal(), 0, status);

	// 이미 생성된 ndf clone 이 있다면 그걸로 NeuroDataFormatProducer를 생성하자!
	if (reader_set.support_ndf && model.SupportNdfClone())
	{
		producer = NeuroDataFormatProducer::GetExistNdfCloneProducer(reader_set, model);

		// ndf clone이 없지만, 사용자가 ndf clone 생성을 원했고, 현재 producer가 ndf clone 생성을 지원한다면 생성하자
		if (!producer)
		{
			JobSignalSender job(reader_set.init_object.GetLongTimeJobSignal(), 0, std::string("creating ndf format clone device(file)").c_str());

			producer = NeuroDataFormatProducer::CreateNewNdfCloneProducer(reader_set, model);
			if (!producer)
				job.failure();
		}
	}

	if (!producer)
	{
		switch (model.GetProducerType())
		{
			case dp::model::_producer_type::image_file:
				producer = new ImageFileProducer(model);
				break;
			case dp::model::_producer_type::mnist_img:
				producer = new MnistImageProducer(model);
				break;
			case dp::model::_producer_type::mnist_label:
				producer = new MnistLabelProducer(model);
				break;
			case dp::model::_producer_type::numeric:
				producer = new NumericProducer(model);
				break;
			case dp::model::_producer_type::nlp:
				producer = new NlpProducer(model);
				break;
		}
	}
	if (!producer)
	{
		creating_producer_job.failure();
		return NULL;
	}
	if (!producer->Create(reader_set))
	{
		delete producer;
		return NULL;
	}
	producer->data_noising = reader_set.data_noising;

	return producer;
}

AbstractProducer::AbstractProducer(const tensor::DataShape& data_shape
	, neuro_size_t start
	, neuron_value scale_min, neuron_value scale_max
	, model::_label_out_type label_out_type)
	: m_start(start), m_data_shape(data_shape), m_data_dim_size(data_shape.GetDimSize())
	, m_scale_min(scale_min), m_scale_max(scale_max)
	, m_label_out_type(label_out_type)
{
	data_noising = false;
}

AbstractProducer::AbstractProducer(const model::AbstractProducerModel& model)
	: AbstractProducer(model.GetDataShape(), model.GetStartPosition(),
		model.GetMinScale(), model.GetMaxScale(), model.GetLabelOutType())
{
	const _neuro_binding_model_set& binding_set = model.GetBindingSet();
	_neuro_binding_model_set::const_iterator it = binding_set.begin();
	for (; it != binding_set.end(); it++)
	{
		_binding_model_type type = (*it)->GetBindingModelType();

		if (type == _binding_model_type::network_input_layer)
			m_preload_buffer.Setup(m_start, m_data_dim_size * sizeof(neuron_value));

		if(type== _binding_model_type::network_output_layer)
			m_label_preload_buffer.Setup(m_start, sizeof(neuro_u32));
	}
}

AbstractProducer::~AbstractProducer()
{
}

bool AbstractProducer::Preload()
{
	if (!SupportPreload())
	{
		m_preload_buffer.Setup(0, 0);
		m_label_preload_buffer.Setup(0, 0);
		return true;
	}
	// 둘중에 최소한 하나는 설정되야 하는데. 왜냐하면, 입/출력 중 하나의 layer에 무조건 연결되어 있어야 하기 때문!
	if (!m_preload_buffer.IsInit() && !m_label_preload_buffer.IsInit())	
		return false;

	DEBUG_OUTPUT(L"start");
	if (m_preload_buffer.GetTotalCount()>0 || m_label_preload_buffer.GetTotalCount()>0)
	{
		DEBUG_OUTPUT(L"already preloaded");
		return true;
	}

	if (GetRawDataCount() == neuro_last64)
	{
		DEBUG_OUTPUT(L"unknown data count");
		return false;
	}

	// 만약 버퍼 크기가 모든 데이터를 포함하면 다 읽어 버리자
	if (!PreloadStart(0))
	{
		DEBUG_OUTPUT(L"failed to read data");
		return false;
	}

	DEBUG_OUTPUT(L"end");
	return true;
}

bool AbstractProducer::PreloadStart(neuro_u64 start_pos)
{
	DEBUG_OUTPUT(L"start. %s", GetTypeString());

	neuro_size_t pos = start_pos + m_start;

	neuro_size_t count = GetRawDataCount();

	bool failed = false;
	if (m_preload_buffer.IsInit())
	{
		if (m_preload_buffer.Resize(count - start_pos))
		{
			for (neuro_size_t pos = start_pos + m_start; pos < count; pos++)
			{
				neuron_value* buffer = (neuron_value*)m_preload_buffer.Get(pos);
				if (!buffer)
				{
					DEBUG_OUTPUT(L"no buffer");
					break;
				}

				neuro_u32 read = ReadRawData(pos, buffer);
				if (read == 0)
				{
					DEBUG_OUTPUT(L"failed to read data");
#ifdef _DEBUG
					// debug test
					ReadRawData(pos, buffer);
#endif
					break;
				}
			}

		}
		else
			failed = true;

	}
	if (m_label_preload_buffer.IsInit())
	{
		if (m_label_preload_buffer.Resize(count - start_pos))
		{
			for (neuro_size_t pos = start_pos + m_start; pos < count; pos++)
			{
				neuro_u32* label = (neuro_u32*)m_label_preload_buffer.Get(pos);
				if (!label)
				{
					DEBUG_OUTPUT(L"no buffer");
					failed = true;
					break;
				}

				neuro_u32 read = ReadRawLabel(pos, *label);
				if (read == 0)
				{
					DEBUG_OUTPUT(L"failed to read data");
#ifdef _DEBUG
					// debug test
					ReadRawLabel(pos, *label);
#endif
					failed = true;
					break;
				}
			}
		}
		else
			failed = true;
	}
	if(!failed)
		OnPreloadCompleted();

	DEBUG_OUTPUT(L"end");
	return true;
}

/*
	ReadRawData는 m_data_dim_size(자기가 읽을 수 있는 것 만큼)만 읽는다.
	그런데 size는 m_data_dim_size 보다 클수 있기 때문에 나머지는 0(또는 -1?)으로 패딩시켜야 한다.
	그런데 ndf 모드에서는 all_fix가 아닌 경우에는 그럴 필요가 없다.
*/
bool AbstractProducer::Read(neuro_size_t pos, neuron_value* buffer)
{
	if (pos + m_start >= GetRawDataCount())
	{
		DEBUG_OUTPUT(L"%s. position[%llu, start:%llu] is over data[%llu]", GetTypeString(), pos, m_start, GetRawDataCount());
		return false;
	}

	// preload buffer는 m_data_dim_size 만큼 데이터가 들어가 있다.
	neuro_u32 read;
	if (m_preload_buffer.IsInit())
		read = m_preload_buffer.Read(pos, buffer) ? m_data_dim_size : 0;
	else
		read = ReadRawData(pos + m_start, buffer);

	if (read == 0)
	{
		DEBUG_OUTPUT(L"%s. position[%llu]. failed to read data", GetTypeString(), pos);
#ifdef _DEBUG
		ReadRawData(pos + m_start, buffer);
#endif
		return false;
	}

	if (read < m_data_dim_size)
	{
//		memset(buffer + m_data_dim_size, 0, sizeof(neuron_value)*(size - m_data_dim_size));
		SetPadding(buffer + read, m_data_dim_size - read);
	}

	if (data_noising)
		DataNoising(buffer);
	return true;
}

bool AbstractProducer::ReadLabel(neuro_size_t pos, neuro_u32& label)
{
	if (pos + m_start >= GetRawDataCount())
	{
		DEBUG_OUTPUT(L"%s. position[%llu, start:%llu] is over data[%llu]", GetTypeString(), pos, m_start, GetRawDataCount());
		return false;
	}

	// preload buffer는 m_data_dim_size 만큼 데이터가 들어가 있다.
	bool ret;
	if (m_label_preload_buffer.IsInit())
		ret = m_label_preload_buffer.Read(pos, &label);
	else
		ret = ReadRawLabel(pos + m_start, label);

	if (!ret)
	{
		DEBUG_OUTPUT(L"%s. position[%llu]. failed to read data", GetTypeString(), pos);
#ifdef _DEBUG
		ReadRawLabel(pos + m_start, label);
#endif
		return false;
	}

	return true;
}

void AbstractProducer::SetPadding(neuron_value* ptr, neuro_u32 size) const
{
	for (neuro_u32 i = 0; i < size; ++i)
		ptr[i] = m_scale_min;
}

MemProducer* AbstractProducer::CreateCloneMemoryProducer() const
{
	// 검증해야함!!
	if (!const_cast<AbstractProducer*>(this)->Preload())
		return NULL;
	/*
	if (m_total_preload_buffer.preload_vector.size() != GetRawDataCount() - m_start)
		return NULL;

	MemProducer* ret = new MemProducer(m_total_preload_buffer.preload_vector.size()*m_data_dim_size, m_data_dim_size);
	if (!ret)
		return NULL;

	const neuro_size_t total_data_dim_size = m_data_dim_size*sizeof(neuron_value);

	neuron_value* p = ret->m_value_buffer.buffer;
	for (neuro_size_t index = 0, n = m_total_preload_buffer.preload_vector.size();index<n;index++)
	{
		const _CACHED_BUFFER& cur = m_total_preload_buffer.preload_vector[index];
		memcpy(p, cur.buf, cur.count*sizeof(neuron_value));
		memset(p + cur.count, 0, (m_data_dim_size - cur.count)*sizeof(neuron_value));

		p += total_data_dim_size;
	}
	return ret;
	*/
	return NULL;
}

MemProducer::MemProducer(neuro_u64 total_size, neuro_u32 column_size)
: AbstractProducer(tensor::DataShape({ column_size == neuro_last32 ? (neuro_u32)total_size : column_size }))
, m_data_count(total_size / m_data_dim_size)
{
	m_value_buffer.Alloc(total_size);
	m_clone_buffer = false;
}

MemProducer::MemProducer(const _VALUE_VECTOR& buf, neuro_u32 column_size)
: AbstractProducer(tensor::DataShape({ column_size == neuro_last32 ? (neuro_u32)buf.count : column_size }))
, m_data_count(buf.count / m_data_dim_size)
{
	m_value_buffer = buf;
	m_clone_buffer = true;
	memcpy(m_value_buffer.buffer, buf.buffer, sizeof(neuron_value)*buf.count);
}

MemProducer::~MemProducer()
{
	if (!m_clone_buffer)
		m_value_buffer.Dealloc();
}

bool MemProducer::SetData(neuron_value* value, neuro_u32 size)
{
	if (!m_value_buffer.buffer)
		return false;

	if (size < m_data_dim_size)
		return false;

	memcpy(m_value_buffer.buffer, value, sizeof(neuron_value)*m_data_dim_size);

	return true;
}


neuro_u32 MemProducer::ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode)
{
	memcpy(value, m_value_buffer.buffer + pos*m_data_dim_size, sizeof(neuron_value)*m_data_dim_size);
	return m_data_dim_size;
}

