#pragma once

#include "DataPreprocessorPropertyConfigure.h"

#include "NeuroData/model/DataProviderModel.h"
#include "NeuroData/model/AbstractProducerModel.h"

#include "desc/TensorShapeDesc.h"

using namespace dp;
using namespace dp::model;

namespace property
{
	class SubProducerPropertyConfigure
	{
	public:
		virtual ~SubProducerPropertyConfigure() {}
		virtual void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) = 0;
		virtual void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const = 0;
	};

	class DataProducerPropertyConfigure : public DataPreprocessorPropertyConfigure<AbstractProducerModel, _producer_type>
	{
	public:
		DataProducerPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model)
			: DataPreprocessorPropertyConfigure(list_ctrl, model)
		{
			m_start_prop = m_labelout_count_prop = m_label_dir_prop = NULL;
		}
		virtual ~DataProducerPropertyConfigure() {}

		_model_property_type GetPropertyType() const override { return _model_property_type::data_producer; }
		std::wstring GetPropertyName() const override
		{
			return L"Data Producer";
		}

		void CompositeProperties() override;
		void PropertyChanged(CModelGridProperty* prop, bool& reload)  const override;

	private:
		const wchar_t* GetSubTypeString(neuro_u32 type) const override
		{
			return ToProducerString((_producer_type)type);
		}

		_producer_type GetModelSubType(const AbstractProducerModel* model) const override
		{
			return model->GetProducerType();
		}

		SubProducerPropertyConfigure* GetConfigure() const;

		CModelGridProperty* m_start_prop;
		CModelGridProperty* m_labelout_count_prop;
		CModelGridProperty* m_label_dir_prop;
	};

	class ImageProducerPropertyConfigure : public SubProducerPropertyConfigure
	{
	public:
		ImageProducerPropertyConfigure();
		virtual ~ImageProducerPropertyConfigure() {}

		enum class _prop_type { none, color_type, red_scale, green_scale, blue_scale, fit_type, width, height};
		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const override;
	};

	class IncreasePredictProducerPropertyConfigure : public SubProducerPropertyConfigure
	{
	public:
		virtual ~IncreasePredictProducerPropertyConfigure() {}

		enum class _prop_type { none, src_ma, src_column, distance, increase_predict_type, predict_range_count, compare_value, compare_type };
		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const override;
	};

	class MnistProducerPropertyConfigure : public SubProducerPropertyConfigure
	{
	public:
		virtual ~MnistProducerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) override {}
		void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const override {}
	};

	class NlpProducerPropertyConfigure : public SubProducerPropertyConfigure
	{
	public:
		virtual ~NlpProducerPropertyConfigure() {}

		enum class _prop_type {
			none, morpheme_parser, mecaprc_path, use_morpheme_type_vector, w2v_path, word_norm
			, parse_sentence, max_sentence, max_word_per_sentence, max_word
			, src_use
		};
		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const override;
	};

	class NumericProducerPropertyConfigure : public SubProducerPropertyConfigure
	{
	public:
		virtual ~NumericProducerPropertyConfigure() {}

		enum class _prop_type { none, onehot, src_ma, src_use, column_index };
		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const override;
	};
}
