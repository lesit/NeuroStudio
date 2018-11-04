
// NeuroStudioDoc.cpp : CNeuroStudioDoc 클래스의 구현
//

#include "stdafx.h"
#include "NeuroStudioApp.h"

#include "NeuroStudioDoc.h"
#include "MainFrm.h"

#include "NeuroStudioView.h"
#include "DeeplearningDesignView.h"
//#include "NeuroSystemView.h"
#include "util/StringUtil.h"
#include "util/FileUtil.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CNeuroStudioDoc

IMPLEMENT_DYNCREATE(CNeuroStudioDoc, CDocument)

BEGIN_MESSAGE_MAP(CNeuroStudioDoc, CDocument)
	ON_COMMAND(ID_FILE_NEW, &CNeuroStudioDoc::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CNeuroStudioDoc::OnFileOpen)
	ON_COMMAND(ID_FILE_SAVE, &CNeuroStudioDoc::OnFileSave)
	ON_COMMAND(ID_FILE_SAVE_AS, &CNeuroStudioDoc::OnFileSaveAs)
	ON_COMMAND(ID_NEURALNETWORK_REPLACE, &CNeuroStudioDoc::OnNeuralNetworkReplace)
END_MESSAGE_MAP()


// CNeuroStudioDoc 생성/소멸

CNeuroStudioDoc::CNeuroStudioDoc()
{
	m_project = NULL;
}

CNeuroStudioDoc::~CNeuroStudioDoc()
{
	delete m_project;
	m_project = NULL;
}

// CNeuroStudioDoc serialization

void CNeuroStudioDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}


// CNeuroStudioDoc 진단

#ifdef _DEBUG
void CNeuroStudioDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CNeuroStudioDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

#ifdef _DEBUG
CString CNeuroStudioDoc::GetTestPath(CString strName)
{
	wchar_t szFilePath[MAX_PATH];
	GetModuleFileName(NULL, szFilePath, _countof(szFilePath));

	std_wstring_vector str_vector;
	util::StringUtil::CharsetTokenizeString(szFilePath, str_vector, L"\\/");

	int i = 0;
	for (; i<str_vector.size(); i++)
	{
		if (wcscmp(str_vector[i].c_str(), L"projects") == 0 && i>0)
		{
			std::wstring proj_path;
			for (int j = 0; j<i; j++)
			{
				proj_path += str_vector[j].c_str();
				proj_path += L"\\";
			}

			proj_path += L"test\\";
			proj_path += strName;
			proj_path += L"\\";
			proj_path += strName;
			proj_path += L".nsystem";
			return proj_path.c_str();
		}
	}
	return L"";
}
#endif

bool CNeuroStudioDoc::CreateNewProject()
{
	DeepLearningDesignViewManager* design_view = NULL;

	POSITION pos = GetFirstViewPosition();
	while (pos)
	{
		CView* pView = GetNextView(pos);
		if (pView->IsKindOf(RUNTIME_CLASS(CNeuroStudioView)))
			int a = 0;
		else if (pView->IsKindOf(RUNTIME_CLASS(DeeplearningDesignView)))
			design_view = (DeepLearningDesignViewManager*)(DeeplearningDesignView*)pView;
	}
	if (design_view == NULL)
		return false;

	delete m_project;
	m_project = new np::project::NeuroStudioProject(*this, *design_view);
	if (m_project == NULL)
		return false;

	SetTitle(L"New Project");

	return true;
}

// CNeuroStudioDoc 명령
BOOL CNeuroStudioDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	if(!CreateNewProject())
		return FALSE;

#ifdef _DEBUG
//	m_project->SampleProject();
//	OnOpenDocument(L"D:\\AI & Human\\test\\Mnist-data\\mnist_cnn_1.0.0.1.nsystem");
#endif

	return TRUE;
}

void CNeuroStudioDoc::OnFileNew()
{
	// question

	// create new project
	if (!CreateNewProject())
	{
		DEBUG_OUTPUT(L"failed");
		return;
	}
}

void CNeuroStudioDoc::OnFileOpen()
{
	TCHAR BASED_CODE szFilter[] = _T("Neuro System (*.nsystem)|*.nsystem|All Files (*.*)|*.*||");

	CFileDialog	dlg(TRUE, _T("*.nsystem"), _T(""), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = _T("AI & Human Neuro System");
	if(dlg.DoModal()!=IDOK)
		return;

	OnOpenDocument(dlg.GetPathName());

	// 자꾸 오픈할때 멈추는게 여기까지 안와서 그런가? 그랬으면 메뉴 클릭도 안될텐데...
}

BOOL CNeuroStudioDoc::OnOpenDocument(LPCTSTR lpszPathName)
{
	if(_tcslen(lpszPathName)==0)
		return FALSE;

	if (!CreateNewProject())
	{
		DEBUG_OUTPUT(L"failed");
		return FALSE;
	}

	HCURSOR oldCursor=SetCursor(LoadCursor(NULL, IDC_WAIT));
	if(!oldCursor)
		oldCursor=LoadCursor(NULL, IDC_ARROW);

	bool bRet = m_project->LoadProject(util::StringUtil::WideToMultiByte(lpszPathName).c_str());

	SetCursor(oldCursor);

	if (!bRet)
		return FALSE;

	SetPathName(lpszPathName, TRUE);

	return TRUE;
}

std::string CNeuroStudioDoc::RequestNewProjectfileName(std::string strRecommended)
{
	TCHAR BASED_CODE szFilter[] = _T("Neuro System (*.nsystem)|*.nsystem|All Files (*.*)|*.*||");

	std::wstring init_path = util::StringUtil::MultiByteToWide(strRecommended);
	CFileDialog	dlg(FALSE, L"*.nsystem", init_path.c_str(), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = _T("AI & Human Neural Network");
	dlg.m_ofn.lpstrDefExt = _T(".nsystem");
	if (dlg.DoModal() != IDOK)
		return "";

	return util::StringUtil::WideToMultiByte((const wchar_t*)dlg.GetPathName());
}

bool CNeuroStudioDoc::RequestSaveProject()
{
	if(!m_project)
		return false;

	std::string file_path=m_project->GetProjectFilePath();
	if(file_path.empty())
		file_path = RequestNewProjectfileName();

	return OnSaveDocument(util::StringUtil::MultiByteToWide(file_path).c_str()) != FALSE;
}

void CNeuroStudioDoc::OnFileSave()
{
	RequestSaveProject();
}

void CNeuroStudioDoc::OnFileSaveAs()
{
	std::string filename = util::FileUtil::GetNameFromFileName<char>(m_project->GetProjectFilePath()).c_str();
	if (!filename.empty())
		filename += "_backup";
	OnSaveDocument(util::StringUtil::MultiByteToWide(RequestNewProjectfileName(filename)).c_str());
}

BOOL CNeuroStudioDoc::OnSaveDocument(LPCTSTR lpszPathName)
{
	if(!m_project)
		return FALSE;

	if (_tcslen(lpszPathName) == 0)
		return FALSE;

	HCURSOR oldCursor=SetCursor(LoadCursor(NULL, IDC_WAIT));
	if(!oldCursor)
		oldCursor=LoadCursor(NULL, IDC_ARROW);

	bool bRet=m_project->SaveProject(util::StringUtil::WideToMultiByte(lpszPathName).c_str());

	SetCursor(oldCursor);

	if (bRet)
		SetPathName(lpszPathName, TRUE);

	return bRet;
}

void CNeuroStudioDoc::OnNeuralNetworkReplace()
{
	if(!m_project)
		return;

	CString strFilePath;
	{
		TCHAR BASED_CODE szFilter[] = _T("Neural Network (*.nsas)|*.nsas|All Files (*.*)|*.*||");

		CFileDialog	dlg(TRUE, _T("*.nsas"), _T(""), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
		dlg.m_ofn.lpstrTitle = _T("AI & Human Neural Network");
		dlg.m_ofn.lpstrDefExt = _T(".nsas");
		if (dlg.DoModal() != IDOK)
			strFilePath = dlg.GetPathName();
	}
	if (strFilePath.IsEmpty())
		return;

	if(!m_project->SaveNetworkStructure(false, false))
		return;

	m_project->OpenNetworkStructure(util::StringUtil::WideToMultiByte((const wchar_t*)strFilePath).c_str());
}


std::string CNeuroStudioDoc::RequestNetworkSavePath(const char* default_name) const
{
	TCHAR BASED_CODE szFilter[] = _T("Neural Network (*.nsas)|*.nsas|All Files (*.*)|*.*||");

	std::wstring init_fname = util::StringUtil::MultiByteToWide(default_name);
	CFileDialog	dlg(FALSE, _T("*.nsas"), init_fname.c_str(), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter);
	dlg.m_ofn.lpstrTitle = _T("AI & Human Neural Network");
	dlg.m_ofn.lpstrDefExt=_T(".nsas");
	if(dlg.DoModal()!=IDOK)
		return "";

	return util::StringUtil::WideToMultiByte((const wchar_t*)dlg.GetPathName());
}

void CNeuroStudioDoc::OnCloseDocument()
{
#ifdef _DEBUG
	if(m_project)
	{
		project::NeuroSystemManager& st=m_project->GetNSManager();
		int a=0;
	}
#endif
	delete m_project;
	m_project = NULL;

	CDocument::OnCloseDocument();
}


void CNeuroStudioDoc::SetTitle(LPCTSTR lpszTitle)
{
	__super::SetTitle(lpszTitle);
}
