#include "stdafx.h"

#include "DataReaderSet.h"

#include "lib/NLPUtil/include/MecabParser.h"
#include "util/FileUtil.h"

#include "NlpProducer.h"

using namespace np;
using namespace np::dp;
using namespace np::dp::preprocessor;

InitShareObject::InitShareObject(JobSignalReciever* long_time_job_signal, const std::string& base_dir)
	: m_base_dir(base_dir)
{
	m_long_time_job_signal = long_time_job_signal;

	default_s2w = NULL;
	w2v = NULL;
}

InitShareObject::~InitShareObject()
{
	for (int i = 0; i < mecab_vector.size(); i++)
		delete mecab_vector[i];

	if (default_s2w)
		delete default_s2w;

	delete w2v;
}

nlp::WordToVector* InitShareObject::CreateW2V(const char* path)
{
	if (w2v && util::FileUtil::ComparePath<char>(path, w2v->GetLoadedPath().c_str()))
		return w2v;

	JobSignalSender job(m_long_time_job_signal, 0, std::string("loading word vector from ").append(path).c_str());

	delete w2v;
	w2v = new nlp::WordToVector;

	DEBUG_OUTPUT(L"start load w2v[%s]", util::StringUtil::MultiByteToWide(path).c_str());
	if (!w2v->Load(path) || w2v->GetDimension() == 0)
	{
		last_error_msg = util::StringUtil::Format<wchar_t>(L"failed load[%s]", util::StringUtil::MultiByteToWide(path).c_str());
		DEBUG_OUTPUT(last_error_msg.c_str());

		delete w2v;
		w2v = NULL;

		job.failure();
		return NULL;
	}
	DEBUG_OUTPUT(L"end load w2v");

	return w2v;
}

nlp::SentenceToWord* InitShareObject::GetS2W(const char* mecabrc_path)
{
	if (mecabrc_path!=NULL)
	{
		for (int i = 0; i < mecab_vector.size(); i++)
		{
			if (util::FileUtil::ComparePath<char>(mecab_vector[i]->GetRcPath(), mecabrc_path))
				return mecab_vector[i];
		}
		return NULL;
	}
	else
		return default_s2w;
}

nlp::SentenceToWord* InitShareObject::CreateS2W(const char* mecabrc_path)
{
	nlp::SentenceToWord* old = GetS2W(mecabrc_path);
	if (old)
		return old;

	if (mecabrc_path)
	{
		std::string a_mecabrc_path; a_mecabrc_path.assign(mecabrc_path, mecabrc_path + strlen(mecabrc_path));
		np::nlp::MecabParser* mecab = new np::nlp::MecabParser(a_mecabrc_path.c_str());
		if (!mecab->Initialize())
		{
			last_error_msg = util::StringUtil::Format<wchar_t>(L"failed initialize mecab[rc path=%s", mecabrc_path);
			DEBUG_OUTPUT(last_error_msg.c_str());
			delete mecab;
			mecab = NULL;
		}
		mecab_vector.push_back(mecab);
		return mecab;
	}
	else
	{
		if (default_s2w)
			return default_s2w;

		default_s2w = new np::nlp::SentenceToWord();
		if (!default_s2w->Initialize())
		{
			last_error_msg = L"failed initialize SentenceToWord";
			DEBUG_OUTPUT(last_error_msg.c_str());

			delete default_s2w;
			default_s2w = NULL;
		}
		return default_s2w;
	}
}

#include "storage/MemoryDeviceAdaptor.h"

DataReaderSet::DataReaderSet(InitShareObject& _init_object, bool _data_noising, bool _support_ndf, neuro_u32 _batch_size)
: init_object(_init_object), data_noising(_data_noising), support_ndf(_support_ndf), batch_size(_batch_size)
{
}

DataReaderSet::~DataReaderSet()
{
	for (_uid_mem_data_map::const_iterator it = memory_data_map.begin(); it != memory_data_map.end(); it++)
		delete it->second.data;

	for(_uid_reader_map::const_iterator it=reader_map.begin();it!=reader_map.end();it++)
		delete it->second;

	for (_uid_producer_map::const_iterator it = producer_map.begin(); it != producer_map.end(); it++)
		delete it->second;
}

bool DataReaderSet::CreateDevices(const model::AbstractPreprocessorModel& model, std::vector<device::DeviceAdaptor*>& device_vector)
{
	if (!model.AvailableAttachDeviceInput())
		return false;

	const DataSourceNameVector<char>* name_vector = GetDataNameVector(model.uid);
	if (name_vector != NULL)
	{
		for (neuro_u32 i = 0, n=name_vector->GetCount(); i < n; i++)
		{
			device::FileDeviceFactory fda(name_vector->GetPath(i).c_str());
			device::DeviceAdaptor* device = fda.CreateReadOnlyAdaptor();
			if (device)
				device_vector.push_back(device);
		}
		return device_vector.size() > 0;
	}

	const _MEM_DATA_SOURCE* mem_data = GetMemoryData(model.uid);
	if (mem_data != NULL)
	{
		device::MemoryDeviceRefAdaptor* device = new device::MemoryDeviceRefAdaptor(mem_data->size, mem_data->data);
		device_vector.push_back(device);
		return device_vector.size() > 0;
	}

	return false;
}

