#pragma once

#include "common.h"
#include "util/JobSignal.h"

#include "storage/DeviceAdaptor.h"

#include "../model/AbstractPreprocessorModel.h"

#include "DataSourceNameVector.h"

namespace np
{
	namespace nlp
	{
		class MecabParser;
		class SentenceToWord;
		class WordToVector;
	}

	namespace dp
	{
		namespace preprocessor
		{
			class InitShareObject
			{
			public:
				InitShareObject(JobSignalReciever* long_time_job_signal, const std::string& base_dir);
				virtual ~InitShareObject();

				JobSignalReciever* GetLongTimeJobSignal()
				{
					return m_long_time_job_signal;
				}

				const char* GetBaseDir() const { return m_base_dir.c_str(); }

				nlp::SentenceToWord* CreateS2W(const char* mecabrc_path);
				nlp::SentenceToWord* GetS2W(const char* mecabrc_path);

				nlp::WordToVector* CreateW2V(const char* path);

				const wchar_t* GetLastErrorMessage() const { return last_error_msg.c_str(); }
			private:
				JobSignalReciever* m_long_time_job_signal;

				const std::string m_base_dir;

				std::vector<nlp::MecabParser*> mecab_vector;
				nlp::SentenceToWord* default_s2w;

				np::nlp::WordToVector* w2v;

				std::wstring last_error_msg;
			};

			struct _MEM_DATA_SOURCE
			{
				neuro_u32 size;
				void* data;
			};
			typedef std::unordered_map<neuro_u32, _MEM_DATA_SOURCE> _uid_mem_data_map;

			class AbstractReader;
			typedef std::unordered_map<neuro_u32, AbstractReader*> _uid_reader_map;

			class AbstractProducer;
			typedef std::unordered_map<neuro_u32, AbstractProducer*> _uid_producer_map;

			class DataReaderSet
			{
			public:
				virtual ~DataReaderSet();

				bool CreateDevices(const model::AbstractPreprocessorModel& model, std::vector<device::DeviceAdaptor*>& device_vector);

				InitShareObject& init_object;

				const bool data_noising;
				const bool support_ndf;

				const neuro_u32 batch_size;

				const DataSourceNameVector<char>* GetDataNameVector(neuro_u32 uid) const
				{
					_uid_datanames_map::const_iterator it = uid_datanames_map.find(uid);
					if( it == uid_datanames_map.end())
						return NULL;
					return &it->second;
				}

				const _MEM_DATA_SOURCE* GetMemoryData(neuro_u32 uid) const
				{
					_uid_mem_data_map::const_iterator it = memory_data_map.find(uid);
					if (it == memory_data_map.end())
						return NULL;
					return &it->second;
				}

				AbstractReader* GetReader(neuro_u32 uid)
				{
					_uid_reader_map::const_iterator it = reader_map.find(uid);
					if (it == reader_map.end())
						return NULL;
					return it->second;
				}

				AbstractProducer* GetProducer(neuro_u32 uid)
				{
					_uid_producer_map::const_iterator it = producer_map.find(uid);
					if (it == producer_map.end())
						return NULL;
					return it->second;
				}

			protected:
				DataReaderSet(InitShareObject& init_object, bool data_noising, bool support_ndf, neuro_u32 batch_size);

				_uid_datanames_map uid_datanames_map;
				_uid_mem_data_map memory_data_map;

				_uid_reader_map reader_map;
				_uid_producer_map producer_map;
			};
		}
	}
}
