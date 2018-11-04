#pragma once

#include "common.h"

#include "NeuroData/model/AbstractPreprocessorModel.h"
#include "NeuroData/model/AbstractReaderModel.h"
#include "NeuroData/model/AbstractProducerModel.h"

namespace np
{
	using namespace dp::model;

	namespace str_rc
	{
		class PreprocessorDesc
		{
		public:
			static wchar_t* GetName(const AbstractPreprocessorModel& model)
			{
				if (model.GetModelType() == _model_type::reader)
				{
					switch (((AbstractReaderModel&)model).GetReaderType())
					{
					case _reader_type::binary:
						return L"Binary reader";
					case _reader_type::text:
						return L"Text reader";
/*					case _reader_type::database:
						return L"DB reader";*/
					}
				}
				else if (model.GetModelType() == _model_type::producer)
				{
					switch (((AbstractProducerModel&)model).GetProducerType())
					{
					case _producer_type::increase_predict:
						return L"Increase Predict";
					case _producer_type::java:
						return L"Java";
					case _producer_type::image_file:
						return L"Image file";
					case _producer_type::mnist_img:
						return L"Mnist image";
					case _producer_type::mnist_label:
						return L"Mnist label";
					case _producer_type::imagenet:
						return L"ImageNet";
					case _producer_type::cifar:
						return L"Cifar";
					case _producer_type::nlp:
						return L"Natural Language";
					case _producer_type::numeric:
						return L"Numeric";
					case _producer_type::windll:
						return L"Windows dll";
					}
				}
				return L"";
			}
		};
	}
}
