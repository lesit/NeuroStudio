#include "stdafx.h"

#include "ImageFileProducer.h"

using namespace np::dp;
using namespace np::dp::model;
using namespace np::dp::preprocessor;

ImageFileProducer::ImageFileProducer(const model::AbstractProducerModel& model)
: AbstractProducer( model), m_model((const ImageFileProducerModel&)model)
, m_img(m_model.GetDefinition().color_type == _color_type::mono ? 1 : 3
	, m_model.GetDefinition().sz.width
	, m_model.GetDefinition().sz.height)
{
	m_img.SetScaleInfo(m_model.GetDefinition().mono_scale
		, model.GetMinScale(), model.GetMaxScale());
}

ImageFileProducer::~ImageFileProducer()
{
}

#include "gui/win32/WinFileUtil.h"

bool ImageFileProducer::Create(DataReaderSet& reader_set)
{
	const DataSourceNameVector<char>* name_vector = reader_set.GetDataNameVector(m_model.uid);
	if (name_vector == NULL || name_vector->GetCount() == 0)
	{
		DEBUG_OUTPUT(L"no files");
		return false;
	}

	const std_string_vector& img_label_dir_vector = m_model.GetLabelDirVector();
	if (img_label_dir_vector.size() > 0)
	{
		std::wstring base_path = util::StringUtil::MultiByteToWide(name_vector->GetPath(0));
		if (base_path.back() != L'\\' || base_path.back() != L'/')
			base_path.push_back(L'\\');

		for (neuro_u32 i = 0, n = img_label_dir_vector.size(); i < n; i++)
		{
			std::wstring path = base_path;
			path.append(util::StringUtil::MultiByteToWide(img_label_dir_vector[i]));
			path.append(L"\\.*");

			std_wstring_vector path_vector;
			gui::win32::WinFileUtil::GetSubFiles(path.c_str(), path_vector);
			if (path_vector.empty())
			{
				DEBUG_OUTPUT(L"%u label. no files in %s", i, path.c_str());
				return false;
			}
			
			for (neuro_u32 i = 0, n = path_vector.size(); i < n; i++)
				m_source_vector.AddPath(path_vector[i].c_str(), i);
		}
	}
	else
	{
		for (neuro_u32 i = 0, n = name_vector->GetCount(); i < n; i++)
			m_source_vector.AddPath(util::StringUtil::MultiByteToWide(name_vector->GetPath(i)).c_str());
	}
	return true;
}

neuro_size_t ImageFileProducer::GetRawDataCount() const
{
	return m_source_vector.GetCount();
}

neuro_u32 ImageFileProducer::ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode)
{
	if (pos >= m_source_vector.GetCount())
		return 0;

	if (!m_img.LoadImage(m_source_vector.GetPath(pos).c_str(), m_model.GetDefinition().fit_type))
		return 0;

	return m_img.ReadData(value, m_data_dim_size) ? m_data_dim_size : 0;
}

bool ImageFileProducer::ReadRawLabel(neuro_size_t pos, neuro_u32& label)
{
	if (pos >= m_source_vector.GetCount())
		return false;

	label = m_source_vector.GetData(pos);
	return true;
}
