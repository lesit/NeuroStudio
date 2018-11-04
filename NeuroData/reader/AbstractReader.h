#pragma once

#include "common.h"

#include "model/AbstractReaderModel.h"
#include "AbstractPreprocessor.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			// NumericFilterProducerDef俊辑 流立 积己茄促
			class AbstractReader : public AbstractPreprocessor
			{
			public:
				static AbstractReader* CreateInstance(DataReaderSet& reader_set, const model::AbstractReaderModel& model);

				AbstractReader() {}
				virtual ~AbstractReader(){};

				model::_model_type GetModelType() const override { return model::_model_type::reader; }

				virtual neuro_u64 GetDataCount() const = 0;
				virtual bool Read(neuro_u64 pos) = 0;
			};
		}
	}
}
