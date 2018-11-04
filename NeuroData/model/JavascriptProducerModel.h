#pragma once

#include "AbstractProducerModel.h"

namespace np
{
	namespace ndr
	{
		namespace model
		{
			class JavascriptProducerModel : public DynamicProducerModel
			{
			public:
				JavascriptProducerModel(neuro_u32 uid);
				virtual ~JavascriptProducerModel();

				virtual _producer_type GetType() const { return _producer_type::javascript; }
				virtual std::wstring GetTypeString() const { return L"java script"; }

				virtual tensor::DataShape GetDataShape() const override { return tensor::DataShape(); }

				static const char* GetHeader();

				const char* GetBody() const;

				bool SetBody(const char* body, std::string& js_err);

				static bool Test(const char* body, std::string& js_err);
			protected:
				static std::string MakeScript(const char* body);

				std::string m_js_body;
			};
		}
	}
}

