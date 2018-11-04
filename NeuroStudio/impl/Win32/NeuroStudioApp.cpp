
// NeuroStudio.cpp : 응용 프로그램에 대한 클래스 동작을 정의합니다.
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
	// 표준 파일을 기초로 하는 문서 명령입니다.
	ON_COMMAND(ID_FILE_NEW, &CWinAppEx::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CWinAppEx::OnFileOpen)
END_MESSAGE_MAP()


// CNeuroStudioApp 생성

CNeuroStudioApp::CNeuroStudioApp()
{

	m_bHiColorIcons = TRUE;

	// TODO: 여기에 생성 코드를 추가합니다.
	// InitInstance에 모든 중요한 초기화 작업을 배치합니다.
}

CNeuroStudioApp::~CNeuroStudioApp()
{
#ifdef _DEBUG
//	_CrtDumpMemoryLeaks();
#endif
}
// 유일한 CNeuroStudioApp 개체입니다.

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

// CNeuroStudioApp 초기화
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

	// 응용 프로그램 매니페스트가 ComCtl32.dll 버전 6 이상을 사용하여 비주얼 스타일을
	// 사용하도록 지정하는 경우, Windows XP 상에서 반드시 InitCommonControlsEx()가 필요합니다. 
	// InitCommonControlsEx()를 사용하지 않으면 창을 만들 수 없습니다.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// 응용 프로그램에서 사용할 모든 공용 컨트롤 클래스를 포함하도록
	// 이 항목을 설정하십시오.
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

	// 표준 초기화
	// 이들 기능을 사용하지 않고 최종 실행 파일의 크기를 줄이려면
	// 아래에서 필요 없는 특정 초기화
	// 루틴을 제거해야 합니다.
	// 해당 설정이 저장된 레지스트리 키를 변경하십시오.
	// TODO: 이 문자열을 회사 또는 조직의 이름과 같은
	// 적절한 내용으로 수정해야 합니다.
	SetRegistryKey(_T("Neuro Studio"));

//	CleanState();	// 레지스트리에 저장된 도킹윈도우 설정을 따르지 않으려 할때

	LoadStdProfileSettings(4);  // MRU를 포함하여 표준 INI 파일 옵션을 로드합니다.

	InitContextMenuManager();

	InitKeyboardManager();

	InitTooltipManager();
	CMFCToolTipInfo ttParams;
	ttParams.m_bVislManagerTheme = TRUE;
	theApp.GetTooltipManager()->SetTooltipParams(AFX_TOOLTIP_TYPE_ALL,
		RUNTIME_CLASS(CMFCToolTipCtrl), &ttParams);

	// 응용 프로그램의 문서 템플릿을 등록합니다. 문서 템플릿은
	//  문서, 프레임 창 및 뷰 사이의 연결 역할을 합니다.
	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CNeuroStudioDoc),
		RUNTIME_CLASS(CMainFrame),       // 주 SDI 프레임 창입니다.
		RUNTIME_CLASS(CNeuroStudioView));
	if (!pDocTemplate)
		return FALSE;
	AddDocTemplate(pDocTemplate);

	// 표준 셸 명령, DDE, 파일 열기에 대한 명령줄을 구문 분석합니다.
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);

	// 명령줄에 지정된 명령을 디스패치합니다.
	// 응용 프로그램이 /RegServer, /Register, /Unregserver 또는 /Unregister로 시작된 경우 FALSE를 반환합니다.
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// 창 하나만 초기화되었으므로 이를 표시하고 업데이트합니다.
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();

	// 접미사가 있을 경우에만 DragAcceptFiles를 호출합니다.
	//  SDI 응용 프로그램에서는 ProcessShellCommand 후에 이러한 호출이 발생해야 합니다.
	return TRUE;
}

int CNeuroStudioApp::ExitInstance()
{
	Gdiplus::GdiplusShutdown(m_gdiplusToken);

	return CWinAppEx::ExitInstance();
}

// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
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

// 대화 상자를 실행하기 위한 응용 프로그램 명령입니다.
void CNeuroStudioApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

// CNeuroStudioApp 사용자 지정 로드/저장 메서드


