#include "stdafx.h"

#include "JavascriptProducerModel.h"
#include "ModuleInterface/JSRunner/JavaScriptRunner.h"

using namespace np::ndr;
using namespace np::ndr::model;

JavascriptProducerModel::JavascriptProducerModel(neuro_u32 uid, const char* body)
	: DynamicProducerModel(uid)
{
	if (body)
		m_js_body = body;
	else
		m_js_body = "var ret = [];\r\n\r\n"
				"if(output_array.length!=4)\r\n"
				"	return ret;\r\n"
				"\r\n"
				"ret = new Array(output_array.length);\r\n"
				"\r\n"
				"// edit contents to make ret array "
				"\r\n"
				"\r\n"
				"return ret";
}

JavascriptProducerModel::~JavascriptProducerModel()
{

}

const char* JavascriptProducerModel::GetHeader()
{
	return "function target(seq, input_array, output_array, param_array)";
}

const char* JavascriptProducerModel::GetBody() const
{
	return m_js_body.c_str();
}

bool JavascriptProducerModel::SetBody(const char* body, std::string& js_err)
{
	m_js_body = body;

	return Test(body, js_err);
}

bool JavascriptProducerModel::Test(const char* body, std::string& js_err)
{
	std::string script = MakeScript(body);

	np::js::JavaScriptRunner* test_script = np::js::JavaScriptRunner::Create(L"./", script.c_str(), js_err);
	if (test_script)
	{
		delete test_script;
		return true;
	}
	else
	{
		return false;
	}
}

std::string JavascriptProducerModel::MakeScript(const char* body)
{
	std::string script = GetHeader();
	script.append("{\n");
	script.append(body);
	script.append("\n}");
	return script;
}
