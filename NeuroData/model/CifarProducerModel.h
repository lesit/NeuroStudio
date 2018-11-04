#pragma once

#include "AbstractProducerModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			class CifarProducerModel : public ImageProcessingProducerModel
			{
			public:
				CifarProducerModel(DataProviderModel& provider, neuro_u32 uid);
				virtual ~CifarProducerModel();
			};
		}
	}
}
