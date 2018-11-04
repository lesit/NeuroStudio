#pragma once

#include "AbstractProducer.h"

#include "model/WindllProducerModel.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			static const char* DynLib_Name_GetDataCount = "GetDataCount";
			static const char* DynLib_Name_ReadData = "ReadData";
			typedef unsigned __int64(_stdcall *DynLib_GetDataCount)();
			typedef bool(_stdcall *DynLib_ReadData)(unsigned __int64 pos, float* buffer, unsigned __int32 size);

			class WindllProducer : public AbstractProducer
			{
			public:
				WindllProducer(const model::AbstractProducerModel& model);
				virtual ~WindllProducer();

				bool Create(DataReaderSet& reader_set) override;

				virtual neuro_size_t GetRawDataCount() const override;
			protected:
				virtual neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode) override;

				const model::WindllProducerModel& m_model;

				HMODULE m_instance;

				DynLib_GetDataCount m_func_GetDataCount;
				DynLib_ReadData m_func_ReadData;
			};
		}
	}
}
