#include "stdafx.h"

#include "NeuroDataFormatProducer.h"
#include "storage/FileDeviceAdaptor.h"

#include "util/FileUtil.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

std::string GetSourceName(DataReaderSet& reader_set, const model::AbstractProducerModel& original_model)
{
	const model::AbstractPreprocessorModel* model = &original_model;
	while (model)
	{
		const model::AbstractPreprocessorModel* input = model->GetInput();
		if (input == NULL)
		{
			const DataSourceNameVector<char>* name_vector = reader_set.GetDataNameVector(model->uid);
			if (name_vector == NULL || name_vector->GetCount()==0)
				return "";	// ndf는 data source가 파일일때만 지원

			std::string ret;
			for (neuro_u32 i = 0, n = name_vector->GetCount(); i < n; i++)
				ret.append(name_vector->GetPath(i)).append("_");

			return ret;
		}

		model = input;
	}
	return "";
}

std::string GetNdfBasePath(DataReaderSet& reader_set, const char* source_name)
{
	std::string ret = reader_set.init_object.GetBaseDir();
	ret.append(source_name);
	return ret;
}

NeuroDataFormatProducer* NeuroDataFormatProducer::GetExistNdfCloneProducer(DataReaderSet& reader_set, const model::AbstractProducerModel& original_model)
{
	std::string source_name = GetSourceName(reader_set, original_model);
	if (source_name.empty())
	{
		DEBUG_OUTPUT(L"no original device name");
		return NULL;
	}

	std::string ndf_path = original_model.MakeNdfPath(GetNdfBasePath(reader_set, source_name.c_str()));
	if (ndf_path.empty())
	{
		DEBUG_OUTPUT(L"no ndf name");
		return NULL;
	}

	DEBUG_OUTPUT(L"ndf device [%s]", util::StringUtil::MultiByteToWide(ndf_path).c_str());

	device::FileDeviceFactory fda(ndf_path.c_str());
	device::DeviceAdaptor* ndf_device = fda.CreateReadOnlyAdaptor();
	if (!ndf_device)	// 파일이 없다.
	{
		DEBUG_OUTPUT(L"no ndf device");
		return NULL;
	}
	// 파일이 있지만, 검증을 해야 한다
	_NDF_HEADER header;
	if (!ReadHeader(*ndf_device, header))
	{
		DEBUG_OUTPUT(L"failed read header");
		goto failed;
	}

	if (!util::FileUtil::ComparePath<char>(header.source_device_name, source_name))
	{
		DEBUG_OUTPUT(L"the orignal device name[%s] in header is not same with [%s]"
			, util::StringUtil::MultiByteToWide(header.source_device_name).c_str()
			, util::StringUtil::MultiByteToWide(source_name).c_str());
		goto failed;
	}

	// 임시용
	header.origin_producer_type = (int)original_model.GetProducerType();	// 실제론 CreateNewNdfCloneProducer에서만 해야함
	{
		tensor::DataShape data_shape;
		header.data_shape.GetVector(const_cast<tensor::DataShape&>(data_shape));

		NeuroDataFormatProducer* producer = new NeuroDataFormatProducer(header, data_shape);
#ifdef _NDF_CHECK_ORG_PRODUCER
		producer->m_check_org_producer = original_model.CreateProducer(source, reader_set);
		if (producer->m_check_org_producer)
			DEBUG_OUTPUT(L"created check org producer");
#endif
		if (!producer->AttachNdfDevice(ndf_device))
		{
			DEBUG_OUTPUT(L"failed producer");
			delete producer;
			goto failed;
		}

		return producer;
	}
failed:
	delete ndf_device;

	return NULL;
}

NeuroDataFormatProducer* NeuroDataFormatProducer::CreateNewNdfCloneProducer(DataReaderSet& reader_set, const model::AbstractProducerModel& original_model)
{
	std::string source_name = GetSourceName(reader_set, original_model);
	if (source_name.empty())
	{
		DEBUG_OUTPUT(L"no original device name");
		return NULL;
	}

	std::string ndf_path = original_model.MakeNdfPath(GetNdfBasePath(reader_set, source_name.c_str()));
	if (ndf_path.empty())
	{
		DEBUG_OUTPUT(L"no ndf name");
		return NULL;
	}

	DEBUG_OUTPUT(L"start. %s -> %s", util::StringUtil::MultiByteToWide(source_name).c_str()
		, util::StringUtil::MultiByteToWide(ndf_path).c_str());

	AbstractProducer* original_producer = AbstractProducer::CreateInstance("creating ndf", reader_set, original_model);
	if (!original_producer)
	{
		DEBUG_OUTPUT(L"failed create original producer");
		return NULL;
	}

	if (original_producer->GetDataCount() == 0)
	{
		DEBUG_OUTPUT(L"no data");
		return NULL;
	}

	device::FileDeviceFactory fda(ndf_path.c_str());
	device::DeviceAdaptor* ndf_device = fda.CreateWriteAdaptor(true, false);
	if (!ndf_device)
	{
		DEBUG_OUTPUT(L"failed create file");
		delete original_producer;
		return NULL;
	}

	std::string status = "creating ndf clone [";
	status.append(ndf_device->GetDeviceName()).append("]");
	JobSignalSender job_status(reader_set.init_object.GetLongTimeJobSignal(), 0, status);

	bool is_completed_create = false;

	neuro_size_t* index_table = NULL;
	neuron_value* read_data_buffer = NULL;

	const tensor::DataShape data_shape = original_model.GetDataShape();

	const neuro_u32 dim_size = data_shape.GetDimSize();
	read_data_buffer = new neuron_value[dim_size];
	if (!read_data_buffer)
	{
		DEBUG_OUTPUT(L"faield alloc read data buffer");
		goto end;
	}

	_NDF_HEADER header;
	memset(&header, 0, sizeof(_NDF_HEADER));
	header.header_size = sizeof(_NDF_HEADER);
	memcpy(header.mark, mark_ndf, sizeof(mark_ndf));
	header.version = ndf_version_0_0_0_2;

	strcpy_s(header.source_device_name, source_name.c_str());
	header.origin_producer_type = (neuro_u32)original_model.GetProducerType();

	header.data_count = original_producer->GetDataCount();

	header.dim_type = original_model.GetNdfDimType();
	header.data_shape.SetVector(data_shape);
	ndf_device->SetPosition(sizeof(_NDF_HEADER));

	neuro_size_t index_table_size = 0;

	const bool isVariableDims = header.dim_type != model::_ndf_dim_type::all_fix;
	if (isVariableDims)
	{
		index_table_size = sizeof(neuro_size_t)*header.data_count;
		ndf_device->SetPosition(ndf_device->GetPosition() + index_table_size);

		index_table = new neuro_size_t[header.data_count];
	}

	neuro_size_t start_data_pos = ndf_device->GetPosition();

	neuro_size_t raw_pos = original_producer->m_start;
	for (neuro_size_t index = 0; index < header.data_count; index++, raw_pos++)
	{
		if (index_table)
			index_table[index] = ndf_device->GetPosition();
		neuro_u32 read = original_producer->ReadRawData(raw_pos, read_data_buffer, isVariableDims);
		if (read == 0 && index != header.data_count - 1)
		{
			DEBUG_OUTPUT(L"failed ReadNDFData");
			goto end;
		}
		if (ndf_device->Write(read_data_buffer, sizeof(neuron_value)*read) != sizeof(neuron_value)*read)
		{
			DEBUG_OUTPUT(L"failed write ndf data");
			goto end;
		}

		header.total_value_count += read;

		if ((index +1) % 1000 == 0)
			job_status.current_status(util::StringUtil::Format(" : creating ndf clone : %llu data", index + 1));
	}

	job_status.current_status(util::StringUtil::Format(" : creating ndf clone : total %llu data", header.data_count));

	neuro_size_t end_data_pos = ndf_device->GetPosition();
	if (header.total_value_count * sizeof(neuron_value) != (end_data_pos - start_data_pos))
	{
		DEBUG_OUTPUT(L"total data size is wrong.");
		goto end;
	}

	ndf_device->SetPosition(0);
	if (ndf_device->Write(&header, sizeof(header)) != sizeof(header))
	{
		DEBUG_OUTPUT(L"failed write header");
		goto end;
	}
	if (isVariableDims)
	{
		if (ndf_device->Write(index_table, index_table_size) != index_table_size)
		{
			DEBUG_OUTPUT(L"failed write index table");
			goto end;
		}
	}

	is_completed_create = true;
	DEBUG_OUTPUT(L"completed create");

end:
	delete ndf_device;
	delete original_producer;
	delete[] read_data_buffer;
	delete[] index_table;

	if (!is_completed_create)
		return NULL;

	return GetExistNdfCloneProducer(reader_set, original_model);
}

bool NeuroDataFormatProducer::ReadHeader(device::DeviceAdaptor& device, _NDF_HEADER& header)
{
	device.SetPosition(0);
	if (device.Read(&header, sizeof(_NDF_HEADER)) != sizeof(_NDF_HEADER))
	{
		DEBUG_OUTPUT(L"failed read header");
		return false;
	}

	neuro_u32 header_size = sizeof(_NDF_HEADER);
	if (header.version == ndf_version_0_0_0_1)
		header_size = ndf_version_0_0_0_1_header_size;

	if (header.header_size != header_size)
	{
		DEBUG_OUTPUT(L"header size is not correct");
		return false;
	}

	neuro_size_t device_total_data_size = device.GetUsageSize() - header.header_size;
	if (header.dim_type != model::_ndf_dim_type::all_fix)	// for index table
		device_total_data_size -= header.data_count * sizeof(neuro_size_t);

	if (header.total_value_count* sizeof(neuron_value) != device_total_data_size)
	{
		DEBUG_OUTPUT(L"header size is not correct");
		return false;
	}

	if (memcmp(header.mark, mark_ndf, sizeof(mark_ndf)) != 0)
	{
		DEBUG_OUTPUT(L"header mark is not correct");
		return false;
	}

	header.source_device_name[_countof(header.source_device_name) - 1] = '\0';

	device.SetPosition(header_size);
	return true;
}

NeuroDataFormatProducer::NeuroDataFormatProducer(const _NDF_HEADER& header, const tensor::DataShape& data_shape)
: AbstractProducer(data_shape)
{
	memcpy(&m_header, &header, sizeof(_NDF_HEADER));

	m_input_device = NULL;
	m_device_size = 0;

	m_index_table = NULL;

	m_device_start_position = 0;
	m_device_cur_position = 0;

	m_last_shape_index = 0;

	m_max_rows = 0;

	m_buffer_size = 0;
	m_read_buffer = NULL;

#ifdef _NDF_CHECK_ORG_PRODUCER
	m_check_org_producer = NULL;
#endif
}

NeuroDataFormatProducer::~NeuroDataFormatProducer()
{
	delete m_input_device;

	if (m_index_table)
		free(m_index_table);

	if (m_read_buffer)
		free(m_read_buffer);

	if (m_header.dim_type!=model::_ndf_dim_type::all_fix)
		DEBUG_OUTPUT(L"max rows in data is %llu", m_max_rows);

#ifdef _NDF_CHECK_ORG_PRODUCER
	if (m_check_org_producer)
		delete m_check_org_producer;
#endif
}

bool NeuroDataFormatProducer::AttachNdfDevice(device::DeviceAdaptor* input_device)
{
	m_input_device = NULL;

	m_device_size = input_device->GetUsageSize();
	if (m_device_size == 0)
	{
		DEBUG_OUTPUT(L"the size is zero");
		return false;
	}

	m_read_data_shape.clear();
	std::wstring read_dim_str;
	for (int i = 0; i < m_data_shape.size(); i++)
	{
		if (m_data_shape[i]>1)
		{
			m_read_data_shape.push_back(m_data_shape[i]);
			read_dim_str.append(util::StringUtil::Transform<wchar_t>(m_data_shape[i])).append(L" ");
		}
	}
	DEBUG_OUTPUT(L"read dim : %s", read_dim_str.c_str());

	m_last_shape_index = m_read_data_shape.size() - 1;
	if (m_header.dim_type == model::_ndf_dim_type::variable_except_last)
		--m_last_shape_index;
	if (m_last_shape_index < 0)
	{
		DEBUG_OUTPUT(L"last_shape index is minus");
		return false;
	}

	const bool isVariableDims = m_header.dim_type != model::_ndf_dim_type::all_fix;
	if (isVariableDims)
	{
		if (m_index_table)
			m_index_table = (neuro_size_t*)realloc(m_index_table, sizeof(neuro_size_t)* m_header.data_count);
		else
			m_index_table = (neuro_size_t*)malloc(sizeof(neuro_size_t)* m_header.data_count);

		DEBUG_OUTPUT(L"read index table : %llu count", m_header.data_count);
		if (input_device->Read(m_index_table, sizeof(neuro_size_t)*m_header.data_count) != sizeof(neuro_size_t)*m_header.data_count)
		{
			DEBUG_OUTPUT(L"failed read index_table");
			return false;
		}
	}
	
	m_device_start_position = m_device_cur_position = input_device->GetPosition();

	if (isVariableDims)
	{
		if (m_device_start_position != m_index_table[0])
		{
			DEBUG_OUTPUT(L"first index is not correct");
			return false;
		}
	}

	if (m_device_size <= m_device_start_position)
	{
		DEBUG_OUTPUT(L"no more data");
		return false;
	}

	m_input_device = input_device;

	m_buffer_size = 10 * 1024 * 1024;
	m_read_buffer = (neuro_u8*)malloc(m_buffer_size);
	if (!m_read_buffer)
	{
		m_buffer_size = 0;
		DEBUG_OUTPUT(L"failed alloc read buffer. size=%llu", m_buffer_size);
		return false;
	}

	// 50mb 이하의 all fix 데이터인 경우 preload 한다.
	if (!isVariableDims && m_header.total_value_count*sizeof(neuron_value) <= 50 * 1024 * 1024)	
	{
		if (PreloadStart(0))
		{
			free(m_read_buffer);
			m_read_buffer = NULL;
			m_buffer_size = 0;
			return true;
		}
		DEBUG_OUTPUT(L"failed preload");
	}
	return true;
}

neuro_u32 NeuroDataFormatProducer::ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode)
{
	if (m_read_buffer == NULL)
	{
		DEBUG_OUTPUT(L"no read buffer");
		return 0;
	}

	const bool isVariableDims = m_header.dim_type != model::_ndf_dim_type::all_fix;

	if (isVariableDims && m_index_table == NULL)
	{
		DEBUG_OUTPUT(L"variable dims. but no index table");
		return 0;
	}
	if (pos >= m_header.data_count)
	{
		DEBUG_OUTPUT(L"position is over data count");
		return 0;
	}

	const neuro_size_t read_position = isVariableDims ? m_index_table[pos] : m_device_start_position + pos*m_data_dim_size*sizeof(neuron_value);
	const neuro_size_t read_end_position =
		isVariableDims ? (pos == m_header.data_count - 1 ? m_device_size : m_index_table[pos + 1]) : read_position + m_data_dim_size*sizeof(neuron_value);

	if (read_end_position <= read_position)
	{
		DEBUG_OUTPUT(L"the read end position[%llu] is under the from positoin[%llu] is over data count", read_end_position, read_position);
		return 0;
	}

	const neuro_size_t data_size = read_end_position - read_position;

	if (m_buffer_size < data_size)
	{
		m_read_buffer = (neuro_u8*)realloc(m_read_buffer, data_size);
		if (m_read_buffer == NULL)
		{
			DEBUG_OUTPUT(L"failed alloc read buffer. %llu size", data_size);
			return 0;
		}

		m_buffer_size = data_size;
	}

	Timer timer;
	float elapse_readfile = 0.f;

	if (m_device_cur_position != read_position)
		m_input_device->SetPosition(read_position);

	neuro_u32 read_size = m_input_device->Read(m_read_buffer, data_size);
	if (read_size < data_size)
	{
		DEBUG_OUTPUT(L"failed read");
		return 0;
	}

	elapse_readfile = timer.elapsed();

	neuron_value* buf_ptr = (neuron_value*)m_read_buffer;
	neuron_value* value_ptr = value;

	ReadRow(buf_ptr, 0, value_ptr);

	/*
	if (timer.elapsed() > 0.01)
	{
		DEBUG_OUTPUT(L"over 0.1 second. position[%llu],%s read size[%u], readfile[%f], readrow[%f]"
			, pos, m_device_cur_position != read_position ? L"random":L"", read_size, elapse_readfile, timer.elapsed());
	}
	*/
	m_device_cur_position = read_position + read_size;

	if (isVariableDims)
		m_max_rows = max(m_max_rows, *((neuro_u32*)m_read_buffer));

	if ((buf_ptr - ((neuron_value*)m_read_buffer))*sizeof(neuro_u32) != read_end_position - read_position)
	{
		DEBUG_OUTPUT(L"read buffer size[%u] is not %u"
			, (buf_ptr - ((neuron_value*)m_read_buffer))*sizeof(neuro_u32), read_end_position - read_position);
		return 0;
	}
	
	if ((value_ptr - value) != m_data_dim_size)
	{
		DEBUG_OUTPUT(L"read value count[%u] is not %u", neuro_u32(value_ptr - value), m_data_dim_size);
		return 0;
	}

#ifdef _NDF_CHECK_ORG_PRODUCER
	if (m_check_org_producer)
	{
		if (m_buffer_size < m_data_dim_size*sizeof(neuron_value))
		{
			m_buffer_size = m_data_dim_size*sizeof(neuron_value);
			if (m_read_buffer)
				m_read_buffer = (neuro_u8*)realloc(m_read_buffer, m_buffer_size);
			else
				m_read_buffer = (neuro_u8*)malloc(m_buffer_size);
			if (!m_read_buffer)
				m_buffer_size = 0;
		}

		neuro_u32 check_read = m_check_org_producer->ReadRawData(pos, (neuron_value*)m_read_buffer, false);
		if (check_read != m_data_dim_size)
		{
			DEBUG_OUTPUT(L"original read count is not equal this");
			check_read = m_check_org_producer->ReadRawData(pos, (neuron_value*)m_read_buffer, false);
			return 0;
		}
		if (memcmp(value, m_read_buffer, m_data_dim_size*sizeof(neuron_value)) != 0)
		{
			DEBUG_OUTPUT(L"failed check with original");
			check_read = m_check_org_producer->ReadRawData(pos, (neuron_value*)m_read_buffer, false);
			return 0;
		}
	}
#endif
	return m_data_dim_size;
}

#include "util/randoms.h"
#include "NlpProducer.h"

void NeuroDataFormatProducer::DataNoising(neuron_value* value)
{
	static bool set_data_post_process = true;
	if (set_data_post_process)
	{
		DEBUG_OUTPUT(L"");
		set_data_post_process = false;
	}

	model::_producer_type origin_producer_type = (model::_producer_type)m_header.origin_producer_type;
	if (origin_producer_type == model::_producer_type::nlp)
	{
		// 스탠포드 대학의 그레이엄롤린슨(Graham Rawlinson)가 문장의 첫 단어와 마지막 단어를 제외하곤 순서 바꿔도 괜찮다고 했음
		neuro_u32 swap_count = 0;

		bool parsing_by_sentence = m_header.origin_producer_info != 0;
		neuron_value* last = value + m_data_dim_size;
		if (!parsing_by_sentence && m_read_data_shape.size() == 2)
		{
			neuro_u32 sub_dim = m_read_data_shape[1];
			neuron_value* start = NlpProducer::FindNotSentenceToken(value, last, sub_dim, m_scale_min);
			if (start == NULL) start = last;
			while (start < last)
			{
				start += sub_dim;	// 문장의 첫단어 제외
				neuron_value* next = NlpProducer::FindSentenceToken(start + sub_dim, last, sub_dim, m_scale_min);
				if (next == start + sub_dim)	// 더이상 없는 것이다.
					break;

				if (next == NULL)
					next = last - sub_dim;

				int length = neuro_float(next - start)/sub_dim - 2;// 문장의 마지막 단어 제외. 형태소 분석을 고려해서 두단어 제외
				if (length>2 && bernoulli(0.5))
				{
					int first_word = uniform_rand(0, length);
					int next_word = uniform_rand(0, length);
					if (first_word != next_word)
					{
						memcpy(m_read_buffer, start + first_word*sub_dim, sub_dim*sizeof(neuron_value));
						memcpy(start + first_word*sub_dim, start + next_word*sub_dim, sub_dim*sizeof(neuron_value));
						memcpy(start + next_word*sub_dim, m_read_buffer, sub_dim*sizeof(neuron_value));

						++swap_count;
					}
				}
				start = next + sub_dim;	// 문장 토큰 제외
			}
		}
		int a = 0;
	}
}

void NeuroDataFormatProducer::ReadRow(neuron_value*& buffer, int depth, neuron_value*& value)
{
	if (m_header.dim_type == model::_ndf_dim_type::all_fix)
	{
		memcpy(value, buffer, sizeof(neuron_value)*m_data_dim_size);
		buffer += m_data_dim_size;
		value += m_data_dim_size;
	}
	else
	{
		neuro_u32 row_count = min(*((neuro_u32*)buffer), m_read_data_shape[depth]);
		++((neuro_u32*&)buffer);

		const neuro_u32 sub_dim_size = m_read_data_shape.GetDimSize(depth + 1);

		if (depth == m_last_shape_index)
		{
			const neuro_u32 copy_count = row_count*sub_dim_size;
			memcpy(value, buffer, sizeof(neuron_value)*copy_count);

			buffer += copy_count;
			value += copy_count;
		}
		else
		{
			for (neuro_u32 column = 0; column < row_count; column++)
				ReadRow(buffer, depth + 1, value);
		}

		neuro_u32 remain = m_read_data_shape[depth] - row_count;
		if (remain > 0)
		{
			const neuro_u32 copy_count = remain*sub_dim_size;
//			memset(value, 0, sizeof(neuron_value)*copy_count);
			SetPadding(value, copy_count);

			value += copy_count;
		}
	}
}
