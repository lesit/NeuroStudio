#pragma once

#include "../model/AbstractPreprocessorModel.h"
#include "DataReaderSet.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class AbstractPreprocessor
			{
			public:
				virtual ~AbstractPreprocessor()
				{
					for (neuro_u32 i = 0; i < m_device_vector.size(); i++)
						delete m_device_vector[i];
				}

				virtual model::_model_type GetModelType() const = 0;

				virtual bool Create(DataReaderSet& reader_set) = 0;

				bool AttachInputDevices(const model::AbstractPreprocessorModel& model, DataReaderSet& reader_set)
				{
					if (model.GetInput() != NULL)
						return false;

					return reader_set.CreateDevices(model, m_device_vector);
				}

			protected:
				std::vector<device::DeviceAdaptor*> m_device_vector;
			};
		}
	}
}
