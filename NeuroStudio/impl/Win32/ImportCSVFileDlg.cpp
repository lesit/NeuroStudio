// DefineCSVSourceDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ImportCSVFileDlg.h"
#include "NeuroDataIO/StringDataFormat.h"

// CImportCSVFileDlg dialog
//IMPLEMENT_DYNAMIC(CImportCSVFileDlg, CDialog)

CImportCSVFileDlg::CImportCSVFileDlg(LPCTSTR strFilePath, NNImportCSVFileManager::_CSV_INFO& csvInfo, CWnd* pParent /*=NULL*/)
	: m_csvInfo(csvInfo), CDialog(CImportCSVFileDlg::IDD, pParent)
{
	m_strFilePath=strFilePath;

	TCHAR szFileName[_MAX_FNAME];
	_tsplitpath(strFilePath, NULL, NULL, szFileName, NULL);
	m_strName=szFileName;
	m_csvInfo.nSkipFirstLine=1;
	m_bReverse=true;

}

CImportCSVFileDlg::~CImportCSVFileDlg()
{
}

void CImportCSVFileDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST_COLUMN, m_ctrColumnList);
	DDX_Control(pDX, IDC_LIST_CSVDATA, m_ctrDataList);
	DDX_Text(pDX, IDC_EDIT_SKIP, m_csvInfo.nSkipFirstLine);
	DDX_Check(pDX, IDC_CHECK_REVERSE, m_bReverse);
	DDX_Text(pDX, IDC_EDIT_NAME, m_strName);
}


BEGIN_MESSAGE_MAP(CImportCSVFileDlg, CDialog)
	ON_BN_CLICKED(IDC_BUTTON_REFRESH, &CImportCSVFileDlg::OnBnClickedButtonRefresh)
END_MESSAGE_MAP()


// CImportCSVFileDlg message handlers

BOOL CImportCSVFileDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	DWORD dwStyle=::GetWindowLong(m_ctrColumnList.GetSafeHwnd(), GWL_STYLE);
	SetWindowLong(m_ctrColumnList.GetSafeHwnd(), GWL_STYLE, dwStyle | LVS_EDITLABELS);
	m_ctrColumnList.SetExtendedStyle(m_ctrColumnList.GetExtendedStyle() | LVS_EX_FULLROWSELECT|LVS_EX_GRIDLINES|LVS_EX_CHECKBOXES );
	m_ctrColumnList.InsertColumn(0, _T("Name"));
	m_ctrColumnList.InsertColumn(1, _T("Type"));

	CStringArray strTypeArray;
	for(neuro_u8 i=0;i<_data_type_count;i++)
		strTypeArray.Add(_data_type_string[i]);

	m_ctrColumnList.AddDefaultProperty(windows::CEditableListCtrl::PIT_STATIC);
	m_ctrColumnList.AddDefaultProperty(strTypeArray);
	m_ctrColumnList.AddDefaultProperty(windows::CEditableListCtrl::PIT_EDIT);

	if(!ahnn::dataio::CSVFileSource::ReadAllData(m_strFilePath, m_rowArray))
	{
		return false;
	}
	if(m_rowArray.size()<2)
	{
		return false;
	}

	for(neuro_u32 iColumn=0;iColumn<m_rowArray[0].size();iColumn++)
	{
		CString strColumn=m_rowArray[0][iColumn].c_str();

		m_ctrColumnList.InsertItem(iColumn, strColumn);
		CString str=m_rowArray[1][iColumn].c_str();
		str.Remove(_T(','));
		str.Remove(_T('+'));

		_data_type type=dataio::DataTypeTest(str);
		m_ctrColumnList.SetItemText(iColumn, 1, _data_type_string[type]);
		m_ctrColumnList.SetItemData(iColumn, (DWORD_PTR)type);
		m_ctrColumnList.SetCheck(iColumn, TRUE);
	}

	m_ctrColumnList.ResizeHeader();

	m_ctrDataList.SetExtendedStyle(m_ctrDataList.GetExtendedStyle() | LVS_EX_GRIDLINES);
	return TRUE;
}

void CImportCSVFileDlg::RefreshDataList()
{
	if(!m_ctrDataList.GetHeaderCtrl())
		return;

	UpdateData(TRUE);

	m_ctrDataList.DeleteAllItems();
	int nDataListColumn=m_ctrDataList.GetHeaderCtrl()->GetItemCount();
	// Delete all of the columns.
	for (int i=0; i < nDataListColumn; i++)
	{
	   m_ctrDataList.DeleteColumn(0);
	}

	CArray<int> checkedArray;
	{
		int nColumn=m_ctrColumnList.GetItemCount();
		int iDataListColumn=0;
		for(int iColumn=0;iColumn<nColumn;iColumn++)
		{
			if(!m_ctrColumnList.GetCheck(iColumn))
				continue;

			CString strColumn=m_ctrColumnList.GetItemText(iColumn, 0);
			m_ctrDataList.InsertColumn(iDataListColumn++, strColumn);

			checkedArray.Add(iColumn);
		}
	}
	if(checkedArray.GetCount()==0)
		return;

	for(neuro_u32 iRow=m_csvInfo.nSkipFirstLine;iRow<m_rowArray.size();iRow++)
	{
		_string_vector& row=m_rowArray[iRow];

		CString strValue=row[checkedArray[0]].c_str();

		int iInsert=m_bReverse ? 0 : m_ctrDataList.GetItemCount();
		iInsert=m_ctrDataList.InsertItem(iInsert, strValue);
		
		int iSubItem=1;
		for(int iCheck=1;iCheck<checkedArray.GetCount();iCheck++)
		{
			strValue=row[checkedArray[iCheck]].c_str();
			m_ctrDataList.SetItemText(iInsert, iSubItem++, strValue);
		}
	}
	for(int i=0;i<checkedArray.GetCount();i++)
		m_ctrDataList.SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);
}

void CImportCSVFileDlg::OnBnClickedButtonRefresh()
{
	RefreshDataList();
}

void CImportCSVFileDlg::OnOK()
{
	UpdateData(TRUE);
	m_csvInfo.strName.assign(m_strName);
	m_csvInfo.bReverse=m_bReverse!=FALSE;

	int nColumn=m_ctrColumnList.GetItemCount();
	for(int iColumn=0;iColumn<nColumn;iColumn++)
	{
		CString strColumn=m_ctrColumnList.GetItemText(iColumn, 0);

		NNImportCSVFileManager::_CSV_INFO::_COLUMN neuroColumn;
		wcscpy(neuroColumn.dataColumn.szName, m_ctrColumnList.GetItemText(iColumn, 0));

		neuroColumn.dataColumn.type=(_data_type)m_ctrColumnList.GetItemData(iColumn);

		neuroColumn.bUse=m_ctrColumnList.GetCheck(iColumn)!=FALSE;
		
		m_csvInfo.columnArray.push_back(neuroColumn);
	}

	CDialog::OnOK();
}
