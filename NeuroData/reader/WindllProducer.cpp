#include "stdafx.h"

#include "WindllProducer.h"
#include "util/StringUtil.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

WindllProducer::WindllProducer(const model::AbstractProducerModel& model)
: AbstractProducer(model), m_model((const model::WindllProducerModel&)model)
{
}

WindllProducer::~WindllProducer()
{
	if (m_instance)
		FreeLibrary(m_instance);
}

bool WindllProducer::Create(DataReaderSet& reader_set)
{
	const DataSourceNameVector<char>* name_vector = reader_set.GetDataNameVector(m_model.uid);
	if (name_vector == NULL || name_vector->GetCount()==0)
		return false;


	m_instance = LoadLibrary(util::StringUtil::MultiByteToWide(name_vector->GetPath(0)).c_str());

	if (m_instance != NULL)
	{
		m_func_GetDataCount = (DynLib_GetDataCount)GetProcAddress(m_instance, DynLib_Name_GetDataCount);
		m_func_ReadData = (DynLib_ReadData)GetProcAddress(m_instance, DynLib_Name_ReadData);
	}
	return true;
}

neuro_size_t WindllProducer::GetRawDataCount() const
{
	if (!m_func_GetDataCount)
		return 0;

	return m_func_GetDataCount();
}

neuro_u32 WindllProducer::ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode)
{
	if (!m_func_ReadData)
		return 0;

	return m_func_ReadData(pos, value, m_data_dim_size) ? m_data_dim_size : 0;
}
