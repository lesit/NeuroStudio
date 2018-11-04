#pragma once

#include "ProjectView/NNDesignViewManager.h"
#include "afxcmn.h"

#include "Windows/EditableListCtrl.h"

using namespace ahnn;
using namespace ahnn::project;
using namespace ahnn::gui;

// CImportCSVFileDlg dialog
class CImportCSVFileDlg : public CDialog
{
public:
//	DECLARE_DYNAMIC(ImportCSVFileDlg)

	CImportCSVFileDlg(LPCTSTR strFilePath, NNImportCSVFileManager::_CSV_INFO& csvInfo, CWnd* pParent = NULL);   // standard constructor
	virtual ~CImportCSVFileDlg();

// Dialog Data
	enum { IDD = IDD_DEFINECSVSOURCEDLG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	virtual BOOL OnInitDialog();
	virtual void OnOK();

	DECLARE_MESSAGE_MAP()
	afx_msg void OnBnClickedButtonRefresh();

protected:
	void RefreshDataList();

private:
	CString m_strFilePath;

	windows::CEditableListCtrl m_ctrColumnList;
	CListCtrl m_ctrDataList;

	NNImportCSVFileManager::_CSV_INFO& m_csvInfo;
	CString m_strName;
	BOOL m_bReverse;

	typedef std::vector<_string_vector> _StringTable;
	_StringTable m_rowArray;;
};

class CImportCSVFileDlgManager : public project::NNImportCSVFileManager
{
public:
	virtual bool ImportCSVFile(const wchar_t* strFilePath, _CSV_INFO& csvInfo)
	{
		CImportCSVFileDlg dlg(strFilePath, csvInfo);
		return dlg.DoModal()==IDOK;
	}
};
