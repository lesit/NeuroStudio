
// NeuroStudio.h : NeuroStudio ���� ���α׷��� ���� �� ��� ����
//
#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"       // �� ��ȣ�Դϴ�.


// CNeuroStudioApp:
// �� Ŭ������ ������ ���ؼ��� NeuroStudio.cpp�� �����Ͻʽÿ�.
//

class CNeuroStudioApp : public CWinAppEx
{
public:
	CNeuroStudioApp();
	virtual ~CNeuroStudioApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.
	BOOL  m_bHiColorIcons;

	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
	virtual int ExitInstance();

private:
	ULONG_PTR	m_gdiplusToken;
};

extern CNeuroStudioApp theApp;
