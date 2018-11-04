#pragma once

#include "common.h"

namespace np
{
	namespace str_rc
	{
		class TensorShapeDesc
		{
		public:
			static std::wstring GetDataShapeText(const np::tensor::DataShape& shape, bool bMultiLine = false);
			static std::wstring GetTensorText(const np::tensor::TensorShape& ts, bool bMultiLine = false);
		};
	}
}

