#include "ImageFileProducerModel.h"

using namespace np::dp;
using namespace np::dp::model;

ImageFileProducerModel::ImageFileProducerModel(DataProviderModel& provider, neuro_u32 uid)
: ImageProcessingProducerModel(provider, uid)
{
}

ImageFileProducerModel::~ImageFileProducerModel()
{
}

void ImageFileProducerModel::SetDefinition(const _NEURO_INPUT_IMAGEFILE_INFO& info)
{
	memcpy(&m_info, &info, sizeof(_NEURO_INPUT_IMAGEFILE_INFO));
	ChangedProperty();
}

void ImageFileProducerModel::SetDefinition(neuro_u32 nWidth, neuro_u32 nHeight, _color_type color_type)
{
	m_info.sz.width=nWidth;
	m_info.sz.height=nHeight;
	m_info.color_type=color_type;
	ChangedProperty();
}

tensor::DataShape ImageFileProducerModel::GetDataShape() const
{
	neuro_u32 channel = 1;
	if (m_info.color_type == _color_type::rgb)
		channel = 3;
	return tensor::DataShape({ channel, (neuro_u32)m_info.sz.height, (neuro_u32)m_info.sz.width });
}
