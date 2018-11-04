#pragma once

#include "DataPreprocessorPropertyConfigure.h"

#include "NeuroData/model/BinaryReaderModel.h"
#include "NeuroData/model/TextReaderModel.h"
#include "NeuroData/model/DataProviderModel.h"

using namespace dp;
using namespace dp::model;

namespace property
{
	class DataReaderPropertyConfigure : public DataPreprocessorPropertyConfigure<AbstractReaderModel, _reader_type>
	{
	public:
		DataReaderPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl, AbstractReaderModel* model)
			: DataPreprocessorPropertyConfigure(list_ctrl, model)
		{}

		virtual ~DataReaderPropertyConfigure() {}

		_model_property_type GetPropertyType() const override { return _model_property_type::data_reader; }
		std::wstring GetPropertyName() const override;

		void CompositeProperties() override;
		void PropertyChanged(CModelGridProperty* prop, bool& reload) const override;

	private:
		enum class _prop_type { none, column_count, skip, reverse, delimiter_type, delimiters, fixed_len, content_token, double_quote, column_token, data_type };

		const wchar_t* GetSubTypeString(neuro_u32 type) const override;
		_reader_type GetModelSubType(const AbstractReaderModel* model) const override;

		bool IsAllowEdit() const
		{
			if (GetModel()->GetReaderType() == _reader_type::text && ((TextReaderModel*)GetModel())->IsImported())
				return false;
			return true;
		}
	};
}
