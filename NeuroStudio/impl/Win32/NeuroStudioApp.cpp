
// NeuroStudio.cpp : ���� ���α׷��� ���� Ŭ���� ������ �����մϴ�.
//

#include "stdafx.h"
#include "afxwinappex.h"
#include "NeuroStudioApp.h"
#include "MainFrm.h"

#include "NeuroStudioDoc.h"
#include "NeuroStudioView.h"

#include "Psapi.h"

#include <gdiplus.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CNeuroStudioApp

BEGIN_MESSAGE_MAP(CNeuroStudioApp, CWinAppEx)
	ON_COMMAND(ID_APP_ABOUT, &CNeuroStudioApp::OnAppAbout)
	// ǥ�� ������ ���ʷ� �ϴ� ���� ����Դϴ�.
	ON_COMMAND(ID_FILE_NEW, &CWinAppEx::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CWinAppEx::OnFileOpen)
END_MESSAGE_MAP()


// CNeuroStudioApp ����

CNeuroStudioApp::CNeuroStudioApp()
{

	m_bHiColorIcons = TRUE;

	// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.
	// InitInstance�� ��� �߿��� �ʱ�ȭ �۾��� ��ġ�մϴ�.
}

CNeuroStudioApp::~CNeuroStudioApp()
{
#ifdef _DEBUG
//	_CrtDumpMemoryLeaks();
#endif
}
// ������ CNeuroStudioApp ��ü�Դϴ�.

CNeuroStudioApp theApp;

#ifdef _DEBUG

#include "ImageTestDlg.h"
void img_test()
{
	CImageTestDlg test;
	test.DoModal();
}

#include "common.h"
#include "core/MemoryManager.h"
#include "util/np_util.h"
#endif

// CNeuroStudioApp �ʱ�ȭ
BOOL CNeuroStudioApp::InitInstance()
{
#ifdef _DEBUG
	{
		core::CUDA_MemoryManager cuda_mem;
		neuro_size_t free, total;
		cuda_mem.GetMemoryInfo(free, total);
		std::string total_str = NP_Util::GetSizeString(total);
		std::string free_str = NP_Util::GetSizeString(free);
		DEBUG_OUTPUT(L"cuda : total(%s), free(%s)\r\n"
			, util::StringUtil::MultiByteToWide(total_str).c_str()
			, util::StringUtil::MultiByteToWide(free_str).c_str());
	}
#endif

//TODO: call AfxInitRichEdit2() to initialize richedit2 library.

	// ���� ���α׷� �Ŵ��佺Ʈ�� ComCtl32.dll ���� 6 �̻��� ����Ͽ� ���־� ��Ÿ����
	// ����ϵ��� �����ϴ� ���, Windows XP �󿡼� �ݵ�� InitCommonControlsEx()�� �ʿ��մϴ�. 
	// InitCommonControlsEx()�� ������� ������ â�� ���� �� �����ϴ�.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// ���� ���α׷����� ����� ��� ���� ��Ʈ�� Ŭ������ �����ϵ���
	// �� �׸��� �����Ͻʽÿ�.
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);
	
	CWinAppEx::InitInstance();

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

	AfxOleInit();

	AfxInitRichEdit2();

	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	Gdiplus::GdiplusStartup(&m_gdiplusToken, &gdiplusStartupInput, NULL);

	// test();

	// ǥ�� �ʱ�ȭ
	// �̵� ����� ������� �ʰ� ���� ���� ������ ũ�⸦ ���̷���
	// �Ʒ����� �ʿ� ���� Ư�� �ʱ�ȭ
	// ��ƾ�� �����ؾ� �մϴ�.
	// �ش� ������ ����� ������Ʈ�� Ű�� �����Ͻʽÿ�.
	// TODO: �� ���ڿ��� ȸ�� �Ǵ� ������ �̸��� ����
	// ������ �������� �����ؾ� �մϴ�.
	SetRegistryKey(_T("Neuro Studio"));

//	CleanState();	// ������Ʈ���� ����� ��ŷ������ ������ ������ ������ �Ҷ�

	LoadStdProfileSettings(4);  // MRU�� �����Ͽ� ǥ�� INI ���� �ɼ��� �ε��մϴ�.

	InitContextMenuManager();

	InitKeyboardManager();

	InitTooltipManager();
	CMFCToolTipInfo ttParams;
	ttParams.m_bVislManagerTheme = TRUE;
	theApp.GetTooltipManager()->SetTooltipParams(AFX_TOOLTIP_TYPE_ALL,
		RUNTIME_CLASS(CMFCToolTipCtrl), &ttParams);

	// ���� ���α׷��� ���� ���ø��� ����մϴ�. ���� ���ø���
	//  ����, ������ â �� �� ������ ���� ������ �մϴ�.
	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CNeuroStudioDoc),
		RUNTIME_CLASS(CMainFrame),       // �� SDI ������ â�Դϴ�.
		RUNTIME_CLASS(CNeuroStudioView));
	if (!pDocTemplate)
		return FALSE;
	AddDocTemplate(pDocTemplate);

	// ǥ�� �� ���, DDE, ���� ���⿡ ���� ������� ���� �м��մϴ�.
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);

	// ����ٿ� ������ ����� ����ġ�մϴ�.
	// ���� ���α׷��� /RegServer, /Register, /Unregserver �Ǵ� /Unregister�� ���۵� ��� FALSE�� ��ȯ�մϴ�.
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// â �ϳ��� �ʱ�ȭ�Ǿ����Ƿ� �̸� ǥ���ϰ� ������Ʈ�մϴ�.
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();

	// ���̻簡 ���� ��쿡�� DragAcceptFiles�� ȣ���մϴ�.
	//  SDI ���� ���α׷������� ProcessShellCommand �Ŀ� �̷��� ȣ���� �߻��ؾ� �մϴ�.
	return TRUE;
}

int CNeuroStudioApp::ExitInstance()
{
	Gdiplus::GdiplusShutdown(m_gdiplusToken);

	return CWinAppEx::ExitInstance();
}

// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()

// ��ȭ ���ڸ� �����ϱ� ���� ���� ���α׷� ����Դϴ�.
void CNeuroStudioApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

// CNeuroStudioApp ����� ���� �ε�/���� �޼���


