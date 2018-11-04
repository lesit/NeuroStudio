#pragma once

#include "../ModelPropertyWnd.h"

#include "NeuroData/model/DataProviderModel.h"

#include "../DesignPreprocessorWnd.h"

using namespace dp;
using namespace dp::model;

namespace property
{
	template<typename Preprocessor, typename _sub_type>
	class DataPreprocessorPropertyConfigure : public ModelPropertyConfigure
	{
	protected:
		DataPreprocessorPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl, Preprocessor* model)
			: ModelPropertyConfigure(list_ctrl)
		{
			m_model = model;
		}

	public:
		virtual ~DataPreprocessorPropertyConfigure()
		{
			std::unordered_map<_sub_type, Preprocessor*>::const_iterator it = m_last_model_map.begin();
			for (; it != m_last_model_map.end(); it++)
				delete it->second;
		}

		virtual void ChangeModel(Preprocessor* model)
		{
			if (m_model != model)
			{
				std::unordered_map<_sub_type, Preprocessor*>::const_iterator it = m_last_model_map.begin();
				for (; it != m_last_model_map.end(); it++)
					delete it->second;
				m_last_model_map.clear();
			}

			m_model = model;
		}

		std::vector<neuro_u32> GetSubTypeVector() const override
		{
			std::vector<_sub_type> type_vector;
			m_model->GetAvailableChangeTypes(type_vector);

			std::vector<neuro_u32> ret;
			ret.resize(type_vector.size());
			for (neuro_u32 i = 0; i < ret.size(); i++)
				ret[i] = (neuro_u32)type_vector[i];
			return ret;
		}

		neuro_u32 GetModelSubType() const override
		{
			return (neuro_u32)GetModelSubType(m_model);
		}

		bool SubTypeChange(AbstractBindedViewManager* view, neuro_u32 type) override
		{
			_sub_type cur_type = GetModelSubType(m_model);
			if (cur_type == (_sub_type)type)
				return false;

			DesignPreprocessorWnd* provider_wnd = (DesignPreprocessorWnd*)view;

			Preprocessor* prev = m_model;
			m_last_model_map[cur_type] = prev;

			std::unordered_map<_sub_type, Preprocessor*>::const_iterator it = m_last_model_map.find((_sub_type)type);
			if (it != m_last_model_map.end())	// 이미 있다면 재사용. 즉, type을 계속 바꿀수 있기 때문에 바로 전에 설정했던걸로 돌린다.
			{
				m_model = it->second;

				m_last_model_map.erase(it);
			}
			else
			{
				m_model = Preprocessor::CreateInstance(m_model->GetProvider(), (_sub_type)type, prev->uid);
			}

			return provider_wnd->ReplacePreprocessorModel(prev, m_model);
		}

	protected:
		virtual _sub_type GetModelSubType(const Preprocessor* model) const = 0;

		Preprocessor* GetModel() const { return m_model; }

	private:
		Preprocessor* m_model;

		// 타입 변경을 할때 속성들을 저장해두기 위해 사용됨. 
		// 즉, 타입 변경 도중 다시 돌아왔을때 전에 설정했던 속성들을 사용할 수 있다.
		std::unordered_map<_sub_type, Preprocessor*> m_last_model_map;
	};
}
