#pragma once

#include "AbstractProducerModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
//			enum class _library_type { javaclass, clib, python };

			class WindllProducerModel : public DynamicProducerModel
			{
			public:
				WindllProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: DynamicProducerModel(provider, uid)
				{}

				virtual ~WindllProducerModel() {};

				virtual _producer_type GetProducerType() const { return _producer_type::windll; }
				virtual std::wstring GetTypeString() const { return L"MS-Windows dll"; }

				void SetDllName(const wchar_t* name) { m_dll_name = name; }
				const wchar_t* GetDllName() const { return m_dll_name.c_str(); }

			protected:
				std::wstring m_dll_name;
			};
		}
	}
}
