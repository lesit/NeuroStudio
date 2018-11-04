#include "stdafx.h"

#include "BinaryReader.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

BinaryReader::BinaryReader(const model::AbstractReaderModel& model)
: m_model((model::BinaryReaderModel&)model)
{
	m_data_count = 0;
	m_position = 0;
}

BinaryReader::~BinaryReader()
{
}

bool BinaryReader::Create(DataReaderSet& reader_set)
{
	return false;
}

bool BinaryReader::Read(neuro_size_t pos)
{
	if (m_position == pos)	// 이미 읽었다.
		return true;

	return false;
}
