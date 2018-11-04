#pragma once

#include "common.h"
#include "AbstractProducerModel.h"
#include "AbstractReaderModel.h"

#include "gui/Win32/Win32Image.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			enum class _color_type { mono, rgb };
			static const wchar_t* _color_type_string[] = { L"Mono", L"RGB" };
			static const wchar_t* ToString(_color_type type)
			{
				if ((int)type >= _countof(_color_type_string))
					return L"";
				return _color_type_string[(int)type];
			}

			struct _NEURO_INPUT_IMAGEFILE_INFO
			{
				_NEURO_INPUT_IMAGEFILE_INFO()
				{
					color_type = _color_type::rgb;
					fit_type = _stretch_type::fit_down;
					sz = { 1024, 768 };
				}

				_color_type color_type;
				gui::_IMAGEDATA_MONO_SCALE_INFO mono_scale;

				_stretch_type fit_type;
				NP_SIZE sz;
			};

			class ImageFileProducerModel : public ImageProcessingProducerModel
			{
			public:
				ImageFileProducerModel(DataProviderModel& provider, neuro_u32 uid);
				virtual ~ImageFileProducerModel();

				_input_source_type GetInputSourceType() const override {
					if(m_input)
						return _input_source_type::none;
					return _input_source_type::imagefile;
				}

//				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::yes; }
				// 나중에 database가 있고 image data 그대로 읽어 들일수 있으면...
				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::no; }

				virtual _producer_type GetProducerType() const { return _producer_type::image_file; }

				virtual tensor::DataShape GetDataShape() const override;

				const _NEURO_INPUT_IMAGEFILE_INFO& GetDefinition() const { return m_info; }
				void SetDefinition(const _NEURO_INPUT_IMAGEFILE_INFO& info);
				void SetDefinition(neuro_u32 nWidth, neuro_u32 nHeight, _color_type color_type);

				_label_out_type GetLabelOutType() const { return _label_out_type::label_dir; }
			private:
				_NEURO_INPUT_IMAGEFILE_INFO m_info;
			};
		}
	}
}

