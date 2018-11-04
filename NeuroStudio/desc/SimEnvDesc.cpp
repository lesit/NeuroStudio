#include "stdafx.h"

#include "SimEnvDesc.h"

using namespace np::str_rc;
const wchar_t* SimEnvDesc::GetDataSetupString(_DATA_SETUP_TYPE type)
{
	const wchar_t* string[] = {
		L"Data setup"
		, L"Train data"
		, L"Run data"
		, L"Start index"
		, L"End index"
		, L"Use train data"
		, L"Next train data"
		, L"start data index of train input data"
		, L"end data index of train input data. 0 means infinite"
		, L"start data index of run input data"
		, L"Use train data in running"
		, L"start data index is next to end data index of train"
	};

	if (type >= _countof(string))
		return L"";

	return string[type];
}

const wchar_t* SimEnvDesc::GetTrainSetupString(_TRAIN_SETUP_TYPE type)
{
	const wchar_t* string[] = {
		L"Train setup"
		, L"Train repeat count"
		, L"Repeat per seq"
		, L"Full Repeat"
		, L"Learn rate"
		, L"Train repeat count per data sequence"
		, L"Train repeat count for all data sequence"
		, L"Learn rate"
	};

	if (type >= _countof(string))
		return L"";
	return string[type];
}

const wchar_t* SimEnvDesc::GetStatSetupString(_RUN_SETUP_TYPE type)
{
	const wchar_t* string[] = {
		L"Output statistics"
		, L"Tolerance"
		, L"tolerance of output value about target value. for example, if this is 0.4, value 0.6 is in tolerance for value 1.0"
	};

	if (type >= _countof(string))
		return L"";
	return string[type];
}

