
// W2VTrainTool.cpp : ���� ���α׷��� ���� Ŭ���� ������ �����մϴ�.
//

#include "stdafx.h"
#include "W2VTrainTool.h"
#include "W2VTrainToolDlg.h"

#include "util/np_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CW2VTrainToolApp

BEGIN_MESSAGE_MAP(CW2VTrainToolApp, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()


// CW2VTrainToolApp ����

CW2VTrainToolApp::CW2VTrainToolApp()
{
	// �ٽ� ���� ������ ����
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.
	// InitInstance�� ��� �߿��� �ʱ�ȭ �۾��� ��ġ�մϴ�.
}


// ������ CW2VTrainToolApp ��ü�Դϴ�.

CW2VTrainToolApp theApp;

#ifdef _DEBUG
#include "TextParsingReader.h"
#endif

BOOL CW2VTrainToolApp::InitInstance()
{
	// ���� ���α׷� �Ŵ��佺Ʈ�� ComCtl32.dll ���� 6 �̻��� ����Ͽ� ���־� ��Ÿ����
	// ����ϵ��� �����ϴ� ���, Windows XP �󿡼� �ݵ�� InitCommonControlsEx()�� �ʿ��մϴ�.
	// InitCommonControlsEx()�� ������� ������ â�� ���� �� �����ϴ�.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// ���� ���α׷����� ����� ��� ���� ��Ʈ�� Ŭ������ �����ϵ���
	// �� �׸��� �����Ͻʽÿ�.
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();

	wchar_t szFilePath[_MAX_PATH];
	GetModuleFileName(AfxGetInstanceHandle(), szFilePath, _MAX_PATH);

	wchar_t szDrv[_MAX_DRIVE];
	wchar_t szDir[_MAX_DIR];
	wchar_t szFileName[_MAX_FNAME];
	_wsplitpath(szFilePath, szDrv, szDir, szFileName, NULL);

	CString strLogPath;
	strLogPath += szDrv;
	strLogPath += szDir;
	strLogPath += szFileName;
	strLogPath += L".log";
	np::NP_Util::SetDebugLogWriteFile(strLogPath);

#if 0
//#if defined(_DEBUG)
	{
		using namespace np::nlp::parser;

		_paragraph sentence_vector = { "aaabbb 123" };
		_sentence new_sentence = "45692 sdf";
		const char* start = new_sentence.c_str();
		const char* last = start + new_sentence.size();
		TextParsingReader::AddSentence(sentence_vector, start, last);

		sentence_vector = { "aaabbb 1a23" };
		new_sentence = "4569234 hgd";
		start = new_sentence.c_str();
		last = start + new_sentence.size();
		TextParsingReader::AddSentence(sentence_vector, start, last);

		sentence_vector = { "aaabbb 23" };
		new_sentence = "45692A34 hgd";
		start = new_sentence.c_str();
		last = start + new_sentence.size();
		TextParsingReader::AddSentence(sentence_vector, start, last);

		std::string text0 = "4.24";
		_paragraph sentence_vector0;
		TextParsingReader::ParseParagraph(".", text0, sentence_vector0);

		std::string text01 = " 4.24 ";
		_paragraph sentence_vector01;
		TextParsingReader::ParseParagraph(".", text01, sentence_vector01);

		std::string text1 = "a1 a2 a3 b1.4.61.b2.b3@b4.b5.b6.b7 c1.c2";
		_paragraph sentence_vector1;
		TextParsingReader::ParseParagraph(".", text1, sentence_vector1);

		std::string text2 = "a1 a2 a3 .b1.4.61.b2.b3@b4.b5.b6.b7 c1.c2";
		_paragraph sentence_vector2;
		TextParsingReader::ParseParagraph(".", text2, sentence_vector2);

		std::string text3 = "a1 a2 a3 .b1.4.61.b2.b3@b4.b5.b6.b7. c1.c2";
		_paragraph sentence_vector3;
		TextParsingReader::ParseParagraph(".", text3, sentence_vector3);

		std::string text4 = "a1 a2 a3 .http://www.abc.co.    b1.4.61.b2.b3@b4.b5.b6.b7. c1.c2";
		_paragraph sentence_vector4;
		TextParsingReader::ParseParagraph(".", text4, sentence_vector4);
		int a = 0;
	}
#endif

	// ��ȭ ���ڿ� �� Ʈ�� �� �Ǵ�
	// �� ��� �� ��Ʈ���� ���ԵǾ� �ִ� ��� �� �����ڸ� ����ϴ�.
	CShellManager *pShellManager = new CShellManager;

	// MFC ��Ʈ���� �׸��� ����ϱ� ���� "Windows ����" ���־� ������ Ȱ��ȭ
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerWindows));

	// ǥ�� �ʱ�ȭ
	// �̵� ����� ������� �ʰ� ���� ���� ������ ũ�⸦ ���̷���
	// �Ʒ����� �ʿ� ���� Ư�� �ʱ�ȭ
	// ��ƾ�� �����ؾ� �մϴ�.
	// �ش� ������ ����� ������Ʈ�� Ű�� �����Ͻʽÿ�.
	// TODO: �� ���ڿ��� ȸ�� �Ǵ� ������ �̸��� ����
	// ������ �������� �����ؾ� �մϴ�.
	SetRegistryKey(_T("W2VTrainTool"));

	CW2VTrainToolDlg dlg;
	m_pMainWnd = &dlg;
	INT_PTR nResponse = dlg.DoModal();
	if (nResponse == IDOK)
	{
		// TODO: ���⿡ [Ȯ��]�� Ŭ���Ͽ� ��ȭ ���ڰ� ������ �� ó����
		//  �ڵ带 ��ġ�մϴ�.
	}
	else if (nResponse == IDCANCEL)
	{
		// TODO: ���⿡ [���]�� Ŭ���Ͽ� ��ȭ ���ڰ� ������ �� ó����
		//  �ڵ带 ��ġ�մϴ�.
	}
	else if (nResponse == -1)
	{
		TRACE(traceAppMsg, 0, "���: ��ȭ ���ڸ� ������ �������Ƿ� ���� ���α׷��� ����ġ �ʰ� ����˴ϴ�.\n");
		TRACE(traceAppMsg, 0, "���: ��ȭ ���ڿ��� MFC ��Ʈ���� ����ϴ� ��� #define _AFX_NO_MFC_CONTROLS_IN_DIALOGS�� ������ �� �����ϴ�.\n");
	}

	// ������ ���� �� �����ڸ� �����մϴ�.
	if (pShellManager != NULL)
	{
		delete pShellManager;
	}

	// ��ȭ ���ڰ� �������Ƿ� ���� ���α׷��� �޽��� ������ �������� �ʰ�  ���� ���α׷��� ���� �� �ֵ��� FALSE��
	// ��ȯ�մϴ�.
	return FALSE;
}

