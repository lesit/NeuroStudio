
// W2VTrainTool.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CW2VTrainToolApp:
// �� Ŭ������ ������ ���ؼ��� W2VTrainTool.cpp�� �����Ͻʽÿ�.
//

class CW2VTrainToolApp : public CWinApp
{
public:
	CW2VTrainToolApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CW2VTrainToolApp theApp;
