#include "stdafx.h"

#include "SimulationStatusControl.h"

#include "SimulationRunningWnd.h"

SimulationStatusControl::SimulationStatusControl(SimulationRunningWnd& displayWnd)
	: m_displayWnd(displayWnd)
{
}

void SimulationStatusControl::GetElapseString(neuro_float elapse, std::wstring& str)
{
	if (elapse < 60)	// 1분보다 작으면
	{
		str = util::StringUtil::Format<wchar_t>(elapse >= 0.0001 ? L"%.4f sec" : L"%f s", elapse);
	}
	else
	{
		neuro_u32 u_el = elapse;
		neuro_u32 day = u_el / 86400;
		u_el %= 86400;
		neuro_u32 hour = u_el / 3600;
		u_el %= 3600;
		neuro_u32 min = u_el / 60;

		str.clear();
		if (elapse >= 86400.f)
			str += util::StringUtil::Format<wchar_t>(L"%u d, ", day);

		u_el %= 60;
		if (elapse >= 3600.f)	// 시간까지만 나오는경우
			str += util::StringUtil::Format<wchar_t>(L"%2u:%2u:%u", hour, min, u_el);
		else
			str += util::StringUtil::Format<wchar_t>(L"%2u m, %.2f s", min, neuro_float(u_el + neuro_float(elapse-neuro_u32(elapse))));
	}
}
