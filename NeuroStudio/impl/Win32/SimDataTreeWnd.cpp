// SimDataSetupWnd.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "SimDataTreeWnd.h"

#include "util/FileUtil.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CSimDataTreeCtrl

CSimDataTreeCtrl::CSimDataTreeCtrl()
{
}

CSimDataTreeCtrl::~CSimDataTreeCtrl()
{
}

BEGIN_MESSAGE_MAP(CSimDataTreeCtrl, CTreeCtrl)
	ON_NOTIFY_REFLECT(TVN_DELETEITEM, OnTvnDeleteItem)
END_MESSAGE_MAP()

void CSimDataTreeCtrl::InitProvider(const dp::model::_producer_model_vector& producer_vector
	, const dp::preprocessor::_uid_datanames_map& uid_datanames_map)
{
	DeleteAllItems();

	if (producer_vector.size() == 0)
		return;

	for (size_t i = 0; i < producer_vector.size(); i++)
	{
		AbstractProducerModel* producer = producer_vector[i];

		CString name = dp::model::_producer_type_string[(neuro_u32)producer->GetProducerType()];
		HTREEITEM hProducer = InsertItem(name, TVI_ROOT);
		_PRODUCER_TREE_DATA* tree_data = new _PRODUCER_TREE_DATA(producer);
		SetItemData(hProducer, (DWORD_PTR)tree_data);

		dp::preprocessor::_uid_datanames_map::const_iterator it_file = uid_datanames_map.find(producer->uid);
		if (it_file != uid_datanames_map.end())
		{
			const DataSourceNameVector<char>& path_vector = it_file->second;

			if (producer->GetLabelOutType() == _label_out_type::label_dir)
			{
				for (neuro_u32 file_index = 0, n = path_vector.GetCount(); file_index < n; file_index++)
					AddDirPath(hProducer, path_vector.GetPath(file_index));
			}
			else
			{
				for (neuro_u32 file_index = 0, n = path_vector.GetCount(); file_index < n; file_index++)
					AddFilePath(hProducer, *tree_data, path_vector.GetPath(file_index));
			}

			Expand(hProducer, TVE_EXPAND);
		}
	}

	SelectItem(GetRootItem());
}

#include "project/NeuroSystemManager.h"

void CSimDataTreeCtrl::AddDataFiles(HTREEITEM hProducer, const DataSourceNameVector<char>& path_vector, bool bClearPrevious)
{
	if (path_vector.GetCount()==0)
		return;

	if (!hProducer)
		return;

	if (bClearPrevious)
		ClearProdocuerFiles(hProducer);

	const dp::model::AbstractProducerModel* producer = ((_PRODUCER_TREE_DATA*)GetItemData(hProducer))->producer_model;

}

void CSimDataTreeCtrl::AddDirPath(HTREEITEM hProducer, const std::string& path)
{
	if (!hProducer)
		return;

	HTREEITEM hChild = InsertItem(util::StringUtil::MultiByteToWide(path).c_str(), hProducer);
	SetItemData(hChild, (DWORD_PTR) new _FILE_TREE_DATA());
}

void CSimDataTreeCtrl::AddFilePath(HTREEITEM hProducer, _PRODUCER_TREE_DATA& tree_data, const std::string& path)
{
	if (!hProducer)
		return;

	std::string basedir = util::FileUtil::GetDirFromPath<char>(path);

	neuro_u32 basedir_uid = tree_data.basedir.GetId(basedir);

	HTREEITEM hChild = InsertItem(util::StringUtil::MultiByteToWide(util::FileUtil::GetFileName<char>(path.c_str())).c_str(), hProducer);
	SetItemData(hChild, (DWORD_PTR) new _FILE_TREE_DATA(basedir_uid));
}

void CSimDataTreeCtrl::GetDataFiles(_uid_datanames_map& data_filepath_map) const
{
	data_filepath_map.clear();

	HTREEITEM hProducer = GetRootItem();
	while(hProducer != NULL)
	{
		_PRODUCER_TREE_DATA* tree_item = (_PRODUCER_TREE_DATA*)GetItemData(hProducer);
		DataSourceNameVector<char>& path_vector = data_filepath_map[tree_item->producer_model->uid];

		GetDataFiles(hProducer, *tree_item, path_vector);

		hProducer = GetNextSiblingItem(hProducer);
	}
}

void CSimDataTreeCtrl::GetDataFiles(HTREEITEM hProducer, const _PRODUCER_TREE_DATA& tree_item, DataSourceNameVector<char>& path_vector) const
{
	HTREEITEM hChild = GetChildItem(hProducer);
	while (hChild)
	{
		_FILE_TREE_DATA* file_item = (_FILE_TREE_DATA*)GetItemData(hChild);

		std::string basedir;
		if (file_item->basedir_id != neuro_last32)
			basedir = tree_item.basedir.GetPath(file_item->basedir_id);

		std::string name = util::StringUtil::WideToMultiByte((const wchar_t*)GetItemText(hChild));
		path_vector.AddPath(basedir.c_str(), name.c_str());

		hChild = GetNextSiblingItem(hChild);
	}
}

void CSimDataTreeCtrl::ClearProdocuerFiles(HTREEITEM hProducer)
{
	HTREEITEM hChild = GetChildItem(hProducer);
	while (hChild)
	{
		HTREEITEM hNext = GetNextSiblingItem(hChild);

		DeleteItem(hChild);

		hChild = hNext;
	}
}

void CSimDataTreeCtrl::OnTvnDeleteItem(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMTREEVIEW pNMTreeView = reinterpret_cast<LPNMTREEVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	if(pNMTreeView->itemOld.hItem)	// old item으로 해야한다!!
	{
		_TREE_DATA* tree_data = (_TREE_DATA*)pNMTreeView->itemOld.lParam;
		if (tree_data)
			delete tree_data;
	}
}

CSimDataTreeCtrl::_tree_item_type CSimDataTreeCtrl::GetItemType(HTREEITEM hItem) const
{
	_TREE_DATA* tree_data = (_TREE_DATA*)GetItemData(hItem);
	if (!tree_data)
		return _tree_item_type::none;

	return tree_data->GetItemType();
}

CSimDataTreeCtrl::_PRODUCER_TREE_DATA* CSimDataTreeCtrl::GetProducerTreeData(HTREEITEM hItem, HTREEITEM& hProducer) const
{
	_TREE_DATA* tree_data = (_TREE_DATA*)GetItemData(hItem);
	if (!tree_data)
		return NULL;

	hProducer = hItem;
	if(tree_data->GetItemType() != _tree_item_type::producer)
	{
		hProducer = NULL;
		HTREEITEM hParent = GetParentItem(hItem);
		while (hParent)
		{
			hProducer = hParent;
			hParent = GetParentItem(hParent);
		}
		if (hProducer == NULL)
			return NULL;

		tree_data = (_TREE_DATA*)GetItemData(hProducer);
	}
	if (tree_data->GetItemType() != _tree_item_type::producer)
		return NULL;

	return (_PRODUCER_TREE_DATA*)tree_data;
}

void CSimDataTreeCtrl::DeleteSelectedItem()
{
	HTREEITEM hItem = GetSelectedItem();
	if (!hItem)
		return;

	_TREE_DATA* tree_data = (_TREE_DATA*)GetItemData(hItem);
	if (!tree_data || tree_data->GetItemType()!= _tree_item_type::file)
		return;

	DeleteItem(hItem);
}

void CSimDataTreeCtrl::MoveSelectedItem(bool bMoveUp)
{
	HTREEITEM hItem=GetSelectedItem();
	if(!hItem)
		return;

	_TREE_DATA* tree_data = (_TREE_DATA*)GetItemData(hItem);
	if (!tree_data || tree_data->GetItemType() != _tree_item_type::file)
		return;

	HTREEITEM hMove=NULL;
	if(bMoveUp)
		hMove=GetPrevSiblingItem(hItem);
	else
		hMove=GetNextSiblingItem(hItem);
	if(!hMove)
		return;

	CString strText=GetItemText(hMove);
	DWORD_PTR data=GetItemData(hMove);

	SetItemText(hMove, GetItemText(hItem));
	SetItemData(hMove, GetItemData(hItem));

	SetItemText(hItem, strText);
	SetItemData(hItem, data);

	SelectItem(hMove);
}

CSimDataTreeWnd::CSimDataTreeWnd()
{

}

CSimDataTreeWnd::~CSimDataTreeWnd()
{

}

#define IDC_SIM_DATA_TREE		WM_USER+1
#define IDC_SIM_DATA_ADD_BTN	WM_USER+2
#define IDC_SIM_DATA_DEL_BTN	WM_USER+3
#define IDC_SIM_DATA_UP_BTN		WM_USER+4
#define IDC_SIM_DATA_DOWN_BTN	WM_USER+5
#define IDC_SIM_DATA_CREATE_SUBDIR_BTN	WM_USER+6
BEGIN_MESSAGE_MAP(CSimDataTreeWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_NOTIFY(TVN_SELCHANGED, IDC_SIM_DATA_TREE, OnTvnSelchanged)
	ON_BN_CLICKED(IDC_SIM_DATA_ADD_BTN, OnBnClickedAddData)
	ON_BN_CLICKED(IDC_SIM_DATA_DEL_BTN, OnBnClickedDelData)
	ON_BN_CLICKED(IDC_SIM_DATA_UP_BTN, OnBnClickedUpData)
	ON_BN_CLICKED(IDC_SIM_DATA_DOWN_BTN, OnBnClickedDownData)
	ON_BN_CLICKED(IDC_SIM_DATA_CREATE_SUBDIR_BTN, OnBnClickedCreateSubdirs)
END_MESSAGE_MAP()

int CSimDataTreeWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	m_ctrTree.Create(WS_VISIBLE | WS_CHILD | WS_BORDER | TVS_FULLROWSELECT | TVS_HASLINES | TVS_LINESATROOT | TVS_SHOWSELALWAYS, CRect(), this, IDC_SIM_DATA_TREE);
	m_ctrAddBtn.Create(_T("Add"), BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_SIM_DATA_ADD_BTN);
	m_ctrDelBtn.Create(_T("Del"), BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_SIM_DATA_DEL_BTN);
	m_ctrUpBtn.Create(_T("Up"), BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_SIM_DATA_UP_BTN);
	m_ctrDownBtn.Create(_T("Down"), BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_SIM_DATA_DOWN_BTN);

	m_ctrCreateSubdirBtn.Create(_T("Create sub dirs"), BS_PUSHBUTTON | WS_CHILD | WS_BORDER, CRect(), this, IDC_SIM_DATA_CREATE_SUBDIR_BTN);
	return 0;
}

void CSimDataTreeWnd::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	CRect rc(0, 0, cx, cy - 5 - 20 - 5);
	m_ctrTree.MoveWindow(rc);

	rc.top = rc.bottom + 5;
	rc.bottom = cy - 5;
	rc.right = rc.left + (cx - 5 * 3) / 4;
	m_ctrAddBtn.MoveWindow(rc);

	rc.MoveToX(rc.right + 5);
	m_ctrDelBtn.MoveWindow(rc);

	rc.MoveToX(rc.right + 5);

	CRect rcCreateSubdir = rc;
	rcCreateSubdir.right = cx;
	m_ctrCreateSubdirBtn.MoveWindow(rcCreateSubdir);

	m_ctrUpBtn.MoveWindow(rc);

	rc.MoveToX(rc.right + 5);
	m_ctrDownBtn.MoveWindow(rc);

}

void CSimDataTreeWnd::OnTvnSelchanged(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMTREEVIEW pNMTreeView = reinterpret_cast<LPNMTREEVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	BOOL bAdd = FALSE;
	BOOL bDel = FALSE;
	BOOL bUpDown = FALSE;

	HTREEITEM hProducer;
	const CSimDataTreeCtrl::_PRODUCER_TREE_DATA* producer = m_ctrTree.GetProducerTreeData(pNMTreeView->itemNew.hItem, hProducer);

	//	stream type에 따라 sel,make 또는 add,del... 둘중 하나를 선택한다.
	if (producer)
	{
		bool bFileItem = pNMTreeView->itemNew.hItem != hProducer;

		if (producer->producer_model->GetLabelOutType() == _label_out_type::label_dir)
		{
			m_ctrAddBtn.SetWindowText(L"Set");
			m_ctrDelBtn.SetWindowText(L"Clear");

			m_ctrCreateSubdirBtn.ShowWindow(SW_SHOW);
			m_ctrUpBtn.ShowWindow(SW_HIDE);
			m_ctrDownBtn.ShowWindow(SW_HIDE);
			bAdd = bFileItem;
		}
		else
		{
			m_ctrAddBtn.SetWindowText(L"Add");
			m_ctrDelBtn.SetWindowText(L"Del");

			m_ctrCreateSubdirBtn.ShowWindow(SW_HIDE);
			m_ctrUpBtn.ShowWindow(SW_SHOW);
			m_ctrDownBtn.ShowWindow(SW_SHOW);

			bAdd = TRUE;
			bUpDown = bFileItem;
		}
		bDel = bFileItem;	// file item
	}

	m_ctrAddBtn.EnableWindow(bAdd);
	m_ctrDelBtn.EnableWindow(bDel);
	m_ctrUpBtn.EnableWindow(bUpDown);
	m_ctrDownBtn.EnableWindow(bUpDown);
}

#include "gui/win32/CheckDirectory.h"
void CSimDataTreeWnd::OnBnClickedAddData()
{
	HTREEITEM hItem = m_ctrTree.GetSelectedItem();

	HTREEITEM hProducer;
	CSimDataTreeCtrl::_PRODUCER_TREE_DATA* tree_data = m_ctrTree.GetProducerTreeData(hItem, hProducer);
	if (!tree_data)
		return;

	if (tree_data->producer_model->GetLabelOutType() == _label_out_type::label_dir)
	{
		DataSourceNameVector<char> path_vector;
		m_ctrTree.GetDataFiles(hProducer, *tree_data, path_vector);

		m_ctrTree.ClearProdocuerFiles(hProducer);

		std::string old_path = path_vector.GetCount() > 0 ? path_vector.GetPath(0) : "";
		std::string new_path = gui::win32::CheckDirectory::BrowserSelectFolder(this, old_path.c_str(), L"Select Base directory of data");
		if (!new_path.empty())
			m_ctrTree.AddDirPath(hProducer, new_path);

		return;
	}

	CString strTitle = L"Select Data";
	CString strFilter;
	if (tree_data->producer_model->GetInputSourceType() ==dp::model::_input_source_type::imagefile)
		strFilter = L"Image files (*.bmp, *.jpg, *.gif, *.png)|*.bmp;*.jpg;*.gif;*.png|All Files (*.*)|*.*||";
	else
		strFilter = L"All Files (*.*)|*.*|";

	CFileDialog	dlg(TRUE, L"*.*", L"", OFN_ALLOWMULTISELECT, strFilter, this);
	dlg.m_ofn.lpstrTitle = strTitle;
	dlg.m_ofn.lpstrDefExt = _T(".*");

	const int c_cMaxFiles = 100 * MAX_PATH;
	const int c_cbBuffSize = c_cMaxFiles + 1;
	TCHAR* ptr = new TCHAR[c_cbBuffSize];
	dlg.m_ofn.lpstrFile = ptr;
	dlg.m_ofn.lpstrFile[0] = 0;
	dlg.m_ofn.nMaxFile = c_cMaxFiles;
	if (dlg.DoModal() != IDOK)
	{
		delete[] ptr;
		return;
	}

	POSITION pos = dlg.GetStartPosition();
	while (pos)
	{
		std::string path = util::StringUtil::WideToMultiByte((const wchar_t*)dlg.GetNextPathName(pos));
		m_ctrTree.AddFilePath(hProducer, *tree_data, path);
	}
	m_ctrTree.Expand(hProducer, TVE_EXPAND);
	delete[] ptr;
}

void CSimDataTreeWnd::OnBnClickedDelData()
{
	m_ctrTree.DeleteSelectedItem();
}

void CSimDataTreeWnd::OnBnClickedUpData()
{
	m_ctrTree.MoveSelectedItem(true);
}

void CSimDataTreeWnd::OnBnClickedDownData()
{
	m_ctrTree.MoveSelectedItem(false);
}

#include "gui/win32/CheckDirectory.h"

void CSimDataTreeWnd::OnBnClickedCreateSubdirs()
{
	HTREEITEM hProducer;
	CSimDataTreeCtrl::_PRODUCER_TREE_DATA* tree_data = m_ctrTree.GetProducerTreeData(m_ctrTree.GetSelectedItem(), hProducer);
	if (!tree_data)
		return;

	if (tree_data->producer_model->GetLabelOutType() != _label_out_type::label_dir)
		return;

	DataSourceNameVector<char> path_vector;
	m_ctrTree.GetDataFiles(hProducer, *tree_data, path_vector);
	if (path_vector.GetCount()==0)
		return;

	gui::win32::CheckDirectory cd;

	std::wstring base_path = util::StringUtil::MultiByteToWide(path_vector.GetPath(0));
	if (base_path.back() != L'\\' || base_path.back() != L'/')
#ifdef _WINDOWS
		base_path.push_back(L'\\');
#else
		base_path.push_back(L'/');
#endif

	const np::std_string_vector& dir_vector = tree_data->producer_model->GetLabelDirVector();
	for (neuro_u32 i = 0; i < dir_vector.size(); i++)
	{
		std::wstring path = base_path;
		path += util::StringUtil::MultiByteToWide(dir_vector[i]);

		cd.CheckDirPath(path.c_str());
	}
}
