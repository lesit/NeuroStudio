// ExtListCtrl.cpp : implementation file
//

#include "stdafx.h"
#include "EditableListCtrl.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

using namespace np::gui::win32;

CEditableListCtrl::CColumnProperty::CColumnProperty()
{
}

CEditableListCtrl::CColumnProperty::CColumnProperty(PROPERTYITEM_TYPE oneType)
{
	AddProperty(oneType);
}

CEditableListCtrl::CColumnProperty::CColumnProperty(const CStringArray& oneComboItemArray)
{
	AddProperty(oneComboItemArray);
}

CEditableListCtrl::CColumnProperty::~CColumnProperty()
{
}

void CEditableListCtrl::CColumnProperty::AddProperty(PROPERTYITEM_TYPE type)
{
	SetSize(GetCount()+1);
	CProperty& info=GetAt(GetCount()-1);

	info.type=type;
}

void CEditableListCtrl::CColumnProperty::AddProperty(const CStringArray& comboItemArray)
{
	SetSize(GetCount()+1);
	CProperty& info=GetAt(GetCount()-1);

	info.type=PIT_COMBO;
	info.comboItemArray.Copy(comboItemArray);
}

const CEditableListCtrl::CProperty* CEditableListCtrl::CColumnProperty::GetProperty(int column) const
{
	if(column<0 || column>=GetCount())
		return NULL;

	return &GetAt(column);
}

CEditableListCtrl::CListDropTarget::CListDropTarget(CEditableListCtrl& listCtrl)
{
	m_cf=RegisterClipboardFormat(listCtrl.GetDragCF());
}

bool CEditableListCtrl::CListDropTarget::GetSelfDrag(CWnd* pWnd, COleDataObject* pDataObject, _LISTCTRL_DRAG_CLIPBOARD& dragInfo)
{
	CEditableListCtrl* pCtrList=(CEditableListCtrl*)pWnd;
	CSharedFile* pDragObject=(CSharedFile*)pDataObject->GetFileData(m_cf);
	if(!pDragObject)
		return false;

	bool bRet = pDragObject->Read(&dragInfo, sizeof(dragInfo))==sizeof(dragInfo) && dragInfo.pPointer==pWnd;
	
	delete pDragObject;
	return bRet;
}

DROPEFFECT CEditableListCtrl::CListDropTarget::OnDragEnter(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point)
{
	_LISTCTRL_DRAG_CLIPBOARD dragInfo;
	if(!GetSelfDrag(pWnd, pDataObject, dragInfo))
		return DROPEFFECT_NONE;

	CEditableListCtrl* pCtrList=(CEditableListCtrl*)pWnd;

	TRACE(_T("list drag enter.\r\n"));

	UINT uFlags;
	int nIndex = pCtrList->HitTest(point, &uFlags);
	pCtrList->DrawInsert(nIndex);

	return DROPEFFECT_MOVE;
}

DROPEFFECT CEditableListCtrl::CListDropTarget::OnDragOver(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point)
{
	_LISTCTRL_DRAG_CLIPBOARD dragInfo;
	if(!GetSelfDrag(pWnd, pDataObject, dragInfo))
		return DROPEFFECT_NONE;

	CEditableListCtrl* pCtrList=(CEditableListCtrl*)pWnd;

	UINT uFlags;
	int nIndex = pCtrList->HitTest(point, &uFlags);
	pCtrList->DrawInsert(nIndex);

	return DROPEFFECT_MOVE;
}

BOOL CEditableListCtrl::CListDropTarget::OnDrop(CWnd* pWnd, COleDataObject* pDataObject, DROPEFFECT dropEffect, CPoint point)
{
	CEditableListCtrl* pCtrList=(CEditableListCtrl*)pWnd;

	_LISTCTRL_DRAG_CLIPBOARD dragInfo;
	if(!GetSelfDrag(pWnd, pDataObject, dragInfo))
		return FALSE;

	UINT uFlags;
	int nDestIndex = pCtrList->HitTest(point, &uFlags);
	pCtrList->MoveItem(dragInfo.iDragItem, nDestIndex);

	return TRUE;
}

void CEditableListCtrl::CListDropTarget::OnDragLeave( CWnd* pWnd )
{
	CEditableListCtrl* pCtrList=(CEditableListCtrl*)pWnd;

	pCtrList->ResetDrawInsert();
}

/////////////////////////////////////////////////////////////////////////////
// CEditableListCtrl
IMPLEMENT_DYNAMIC(CEditableListCtrl, CListCtrl)

CEditableListCtrl::CEditableListCtrl()
{
	m_nLast=-1;

	m_pDropTarget=NULL;
	m_cf=0;
}

CEditableListCtrl::~CEditableListCtrl()
{
	delete m_pDropTarget;
}

#define IDC_PROPCMBBOX   712
#define IDC_PROPEDITBOX  713
#define IDC_PROPBTNCTRL  714
#define IDC_NAMEEDITBOX  715

BEGIN_MESSAGE_MAP(CEditableListCtrl, CListCtrl)
	//{{AFX_MSG_MAP(CEditableListCtrl)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_NOTIFY_REFLECT(NM_CLICK, OnNMClick)
	ON_NOTIFY_REFLECT(LVN_BEGINDRAG, OnBegindrag)
	ON_NOTIFY_REFLECT(LVN_DELETEITEM, OnDeleteitem)
	ON_CBN_KILLFOCUS(IDC_PROPCMBBOX, OnKillfocusCmbBox)
	ON_CBN_SELCHANGE(IDC_PROPCMBBOX, OnSelchangeCmbBox)
	ON_EN_KILLFOCUS(IDC_PROPEDITBOX, OnKillfocusEditBox)
	ON_BN_CLICKED(IDC_PROPBTNCTRL, OnButton)
	//}}AFX_MSG_MAP
	ON_WM_NCLBUTTONDOWN()
	ON_NOTIFY_REFLECT(LVN_INSERTITEM, &CEditableListCtrl::OnLvnInsertitem)
	ON_WM_SIZE()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CEditableListCtrl message handlers
void CEditableListCtrl::PreSubclassWindow()
{
	SetExtendedStyle(GetExtendedStyle() | LVS_EX_GRIDLINES | LVS_EX_FULLROWSELECT | LVS_EX_INFOTIP);
	
	if(m_cf==0)
		m_cf=RegisterClipboardFormat(GetDragCF());

	if(m_pDropTarget==NULL)
	{
		m_pDropTarget=CreateDropTarget();
		if(m_pDropTarget)
			m_pDropTarget->Register(this);
	}

	__super::PreSubclassWindow();
}

int CEditableListCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	// CG: This line was added by the Palette Support component
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	m_cf=RegisterClipboardFormat(GetDragCF());
	m_pDropTarget=CreateDropTarget();
	if(m_pDropTarget)
		m_pDropTarget->Register(this);
	return 0;
}

void CEditableListCtrl::OnDestroy()
{
	if(m_pDropTarget)
		m_pDropTarget->Revoke();

	__super::OnDestroy();
}

void CEditableListCtrl::AddDefaultProperty(PROPERTYITEM_TYPE type)
{
	m_defaultProperty.AddProperty(type);
}

void CEditableListCtrl::AddDefaultProperty(const CStringArray& comboItemArray)
{
	m_defaultProperty.AddProperty(comboItemArray);
}

int CEditableListCtrl::InsertPropertyItem(int iInsert, LPCTSTR strLabel, CColumnProperty* pPropItem, DWORD_PTR data)
{
	iInsert=__super::InsertItem(iInsert, strLabel);

	_PROPERTY_LIST_ITEM* prop_item=new _PROPERTY_LIST_ITEM;
	prop_item->data=data;
	prop_item->prop=pPropItem;
	__super::SetItemData(iInsert, (DWORD_PTR) prop_item);

	if(pPropItem)
	{
		for(int i=0;i<pPropItem->GetCount();i++)
		{
			CProperty& prop=pPropItem->GetAt(i);
			if(prop.type==PIT_COMBO)
				SetItemText(iInsert, i, prop.comboItemArray[0]);
		}
	}
	return iInsert;
}

int CEditableListCtrl::InsertItem(int iInsert, LPCTSTR strLabel, DWORD_PTR data)
{
	return InsertPropertyItem(iInsert, strLabel, NULL, data);
}

bool CEditableListCtrl::SetItemData(int iItem, DWORD_PTR data)
{
	_PROPERTY_LIST_ITEM* prop_item=(_PROPERTY_LIST_ITEM*) __super::GetItemData(iItem);
	if(!prop_item)
		return false;

	prop_item->data=data;
	return true;
}

DWORD_PTR CEditableListCtrl::GetItemData(int iItem) const
{
	const _PROPERTY_LIST_ITEM* prop_item=(const _PROPERTY_LIST_ITEM*) __super::GetItemData(iItem);
	if(!prop_item)
		return NULL;

	return prop_item->data;
}

void CEditableListCtrl::MoveItem(int iItem, int iTarget)
{
	if(iTarget<0)
		iTarget=GetItemCount();

	if (iItem < iTarget)	// 만약에 방금 삭제한 곳이 이동하려는 곳보다 먼저이면
		--iTarget;

	if(iItem==iTarget)
		return;

	bool bSelected=iItem==GetSelectedItem();

	int nText=GetHeaderCtrl()->GetItemCount();
	CStringArray strBufArray;
	CArray<LPCTSTR> labelArray;
	for(int i=0;i<nText;i++)
	{
		LVITEM item;
		memset(&item, 0, sizeof(LVITEM));
		item.mask=LVIF_NORECOMPUTE|LVIF_TEXT;
		item.iItem=iItem;
		item.iSubItem=i;
		GetItem(&item);

		if(item.pszText!=LPSTR_TEXTCALLBACK)
		{
			strBufArray.Add(GetItemText(iItem, i));	// string 포인터를 담기 위해서 사용함
			item.pszText=(LPTSTR)(LPCTSTR)strBufArray[strBufArray.GetCount()-1];
		}
		labelArray.Add(item.pszText);
	}

	DWORD_PTR data=__super::GetItemData(iItem);

	_PROPERTY_LIST_ITEM* prop_item=(_PROPERTY_LIST_ITEM*)data;
	__super::SetItemData(iItem, NULL);
	DeleteItem(iItem);

	iTarget=__super::InsertItem(iTarget, labelArray[0]);
	__super::SetItemData(iTarget, data);
	for(int i=1;i<nText;i++)
		SetItemText(iTarget, i, labelArray[i]);

	TRACE(L"move(%d, %d) : %X\r\n", iItem, iTarget, data);

	if(bSelected)
		SelectItem(iTarget);
}

const CEditableListCtrl::CProperty* CEditableListCtrl::GetProperty(int iItem, int iSubItem) const
{
	if(iItem<0 || iSubItem<0)
		return NULL;

	const _PROPERTY_LIST_ITEM* prop_item=(const _PROPERTY_LIST_ITEM*) __super::GetItemData(iItem);
	if(prop_item->prop)
		return prop_item->prop->GetProperty(iSubItem);

	return m_defaultProperty.GetProperty(iSubItem);
}

bool CEditableListCtrl::OnChangedProp(int iItem, int iSubItem, LPCTSTR newStr)
{
	if(iItem<0 || iSubItem<0)
		return false;

	SetItemText(iItem, iSubItem, newStr);
	ResizeHeader();
	return true;
}

// 항목을 클릭했을때 항목에 해당하는 control을 띄어 주자.
void CEditableListCtrl::OnNMClick(NMHDR* pNMHDR, LRESULT* pResult)
{
	NM_LISTVIEW* pNMListView = (NM_LISTVIEW*)pNMHDR;

	*pResult = 0;

	TRACE(L"CEditableListCtrl::OnNMClick\r\n");

	EditItem(pNMListView->iItem, pNMListView->iSubItem);
}

void CEditableListCtrl::EditItem(int iItem, int iSubItem)
{
	const CProperty* prop = GetProperty(iItem, iSubItem);
	if (!prop)
		return;

	m_stProp.m_iItem=iItem;
	m_stProp.m_iSubItem=iSubItem;

	CString lBoxSelText=GetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem);

	CRect rect;
	GetSubItemRect(m_stProp.m_iItem, m_stProp.m_iSubItem, LVIR_BOUNDS, rect);
	rect.right=rect.left+GetColumnWidth(m_stProp.m_iSubItem);
	if (prop->type==PIT_COMBO)
	{
		//display the combo box.  If the combo box has already been
		//created then simply move it to the new location, else create it
		rect.top -= 4;
		if(::IsWindow(m_stProp.m_cmbBox.GetSafeHwnd()))
			m_stProp.m_cmbBox.MoveWindow(rect);
		else
		{	
			rect.bottom += 100;
			m_stProp.m_cmbBox.Create(CBS_DROPDOWNLIST | WS_VISIBLE | WS_CHILD | WS_BORDER | CBS_NOINTEGRALHEIGHT
				, rect,this,IDC_PROPCMBBOX);

			if(GetFont())
				m_stProp.m_cmbBox.SetFont(GetFont());
		}

		//add the choices for this particular property
		const CStringArray& comboItemArray=prop->comboItemArray;
		
		m_stProp.m_cmbBox.ResetContent();
		for(INT_PTR iCombo=0;iCombo<comboItemArray.GetCount();iCombo++)
		{
			m_stProp.m_cmbBox.AddString(comboItemArray[iCombo]);
		}

		if(m_stProp.m_cmbBox.GetCount()<=0){
			m_stProp.m_iItem=-1;
			return;
		}

		m_stProp.m_cmbBox.ShowWindow(SW_SHOW);
		m_stProp.m_cmbBox.SetFocus();

		//jump to the property's current value in the combo box
		int j = m_stProp.m_cmbBox.FindStringExact(0,lBoxSelText);
		if(j != CB_ERR)
			m_stProp.m_cmbBox.SetCurSel(j);
		else
			m_stProp.m_cmbBox.SetCurSel(0);

		m_stProp.m_cmbBox.ShowDropDown();
	}
	else if(prop->type==PIT_EDIT)
	{
		rect.top -= 1;
		rect.bottom -= 1;
		if(m_stProp.m_editBox)
			m_stProp.m_editBox.MoveWindow(rect);
		else
		{	
			m_stProp.m_editBox.Create(ES_LEFT | ES_AUTOHSCROLL | WS_VISIBLE | WS_CHILD | WS_BORDER,
							rect,this,IDC_PROPEDITBOX);
			if(GetFont())
				m_stProp.m_editBox.SetFont(GetFont());
		}

		m_stProp.m_editBox.ShowWindow(SW_SHOW);
		m_stProp.m_editBox.SetFocus();
		//set the text in the edit box to the property's current value
		m_stProp.m_editBox.SetWindowText(lBoxSelText);
		m_stProp.m_editBox.SetSel(0, -1);
	}
	else if(prop->type==PIT_STATIC)	// static은 아무일도 시키지 말자.
	{
	}
	else
	{
		/*
		if(rect.Width() > 25)
			rect.left = rect.right - 25;
		rect.bottom -= 3;

		if (m_stProp.m_btnCtrl)
			m_stProp.m_btnCtrl.MoveWindow(rect);
		else
		{	
			m_stProp.m_btnCtrl.Create(_T("..."),BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD,
							rect,this,IDC_PROPBTNCTRL);
			if(GetFont())
				m_stProp.m_btnCtrl.SetFont(GetFont());
		}

		m_stProp.m_btnCtrl.ShowWindow(SW_SHOW);
		m_stProp.m_btnCtrl.SetFocus();
		*/
		OnButton();
	}
}

void CEditableListCtrl::OnKillfocusCmbBox() 
{
	if(!::IsWindow(m_stProp.m_cmbBox.GetSafeHwnd()))
		return;

	m_stProp.m_cmbBox.ShowWindow(SW_HIDE);
	m_stProp.m_iItem=-1;
}

void CEditableListCtrl::OnSelchangeCmbBox()
{
	if(m_stProp.m_iItem<0 || !::IsWindow(m_stProp.m_cmbBox.GetSafeHwnd()))
		return;

	CString selStr;

	m_stProp.m_cmbBox.GetLBText(m_stProp.m_cmbBox.GetCurSel(),selStr);
	
	if(!GetProperty(m_stProp.m_iItem, m_stProp.m_iSubItem))
		return;

	if(OnChangingProp(m_stProp.m_iItem, m_stProp.m_iSubItem, selStr))
	{
		SetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem, selStr);
		OnChangedProp(m_stProp.m_iItem, m_stProp.m_iSubItem, selStr);
	}
	else
		m_stProp.m_cmbBox.SelectString(0, GetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem));

	m_stProp.m_cmbBox.ShowWindow(SW_HIDE);
	SetFocus();
}

void CEditableListCtrl::OnKillfocusEditBox()
{
	if(m_stProp.m_iItem<0 || !::IsWindow(m_stProp.m_editBox.GetSafeHwnd()))
		return;

	CString newStr;
	m_stProp.m_editBox.GetWindowText(newStr);

	if(GetProperty(m_stProp.m_iItem, m_stProp.m_iSubItem))
	{
		if(OnChangingProp(m_stProp.m_iItem, m_stProp.m_iSubItem, newStr))
		{
			SetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem, newStr);
			OnChangedProp(m_stProp.m_iItem, m_stProp.m_iSubItem, newStr);
		}
	}

	m_stProp.m_editBox.ShowWindow(SW_HIDE);

	m_stProp.m_iItem=-1;
	m_stProp.m_iSubItem=-1;
}

void CEditableListCtrl::OnButton()
{
	if(m_stProp.m_iItem<0)
		return;

	const CProperty* prop=GetProperty(m_stProp.m_iItem, m_stProp.m_iSubItem);
	if(!prop)
		return;

	//display the appropriate common dialog depending on what type
	//of chooser is associated with the property
	if (prop->type == PIT_COLOR)
	{
		COLORREF initClr;
		CString currClr = GetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem);
		//parse the property's current color value
		if (currClr.Find(_T("RGB")) > -1)
		{
			int j = currClr.Find(_T(','),3);
			CString bufr = currClr.Mid(4,j-4);
			int RVal = _wtoi(bufr);
			int j2 = currClr.Find(_T(','),j+1);
			bufr = currClr.Mid(j+1,j2-(j+1));
			int GVal = _wtoi(bufr);
			int j3 = currClr.Find(_T(')'),j2+1);
			bufr = currClr.Mid(j2+1,j3-(j2+1));
			int BVal = _wtoi(bufr);
			initClr = RGB(RVal,GVal,BVal);
		}
		else
			initClr = 0;
		
		CColorDialog ClrDlg(initClr);
		
		if (IDOK == ClrDlg.DoModal())
		{
			if(m_stProp.m_btnCtrl.GetSafeHwnd())
				m_stProp.m_btnCtrl.ShowWindow(SW_HIDE);

			COLORREF selClr = ClrDlg.GetColor();
			CString clrStr;
			clrStr.Format(_T("RGB(%d,%d,%d)"),GetRValue(selClr),
						GetGValue(selClr),GetBValue(selClr));

			if(OnChangingProp(m_stProp.m_iItem, m_stProp.m_iSubItem, clrStr)){
				SetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem, clrStr);
				OnChangedProp(m_stProp.m_iItem, m_stProp.m_iSubItem, clrStr);
			}
		}
	}
	else if (prop->type == PIT_FILE)
	{
		CString SelectedFile; 
		CString Filter(_T("Gif Files (*.gif)|*.gif||"));
	
		CFileDialog FileDlg(TRUE, NULL, NULL, NULL,
			Filter);
		
		CString currPath = GetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem);
		FileDlg.m_ofn.lpstrTitle = _T("Select file");
		if (currPath.GetLength() > 0)
			FileDlg.m_ofn.lpstrInitialDir = currPath.Left(
				currPath.GetLength() - currPath.ReverseFind(_T('\\')));

		if(IDOK == FileDlg.DoModal())
		{
			if(m_stProp.m_btnCtrl.GetSafeHwnd())
				m_stProp.m_btnCtrl.ShowWindow(SW_HIDE);

			SelectedFile = FileDlg.GetPathName();
		
			if(OnChangingProp(m_stProp.m_iItem, m_stProp.m_iSubItem, SelectedFile)){
				SetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem, SelectedFile);
				OnChangedProp(m_stProp.m_iItem, m_stProp.m_iSubItem, SelectedFile);
			}
		}
	}
	else if (prop->type == PIT_FONT)
	{	
		CFontDialog FontDlg(NULL,CF_EFFECTS | CF_SCREENFONTS,NULL,this);
		
		if(IDOK == FontDlg.DoModal())
		{
			if(m_stProp.m_btnCtrl.GetSafeHwnd())
				m_stProp.m_btnCtrl.ShowWindow(SW_HIDE);

			CString faceName = FontDlg.GetFaceName();
			
			if(OnChangingProp(m_stProp.m_iItem, m_stProp.m_iSubItem, faceName)){
				SetItemText(m_stProp.m_iItem, m_stProp.m_iSubItem, faceName);
				OnChangedProp(m_stProp.m_iItem, m_stProp.m_iSubItem, faceName);
			}
		}
	}
	else if(prop->type==PIT_BUTTON)
	{
		OnUserButton(m_stProp.m_iItem, m_stProp.m_iSubItem);
	}

	m_stProp.m_iItem=-1;
	m_stProp.m_iSubItem=-1;
}

void CEditableListCtrl::OnDeleteitem(NMHDR* pNMHDR, LRESULT* pResult) 
{
	NM_LISTVIEW* pNMListView = (NM_LISTVIEW*)pNMHDR;

	_PROPERTY_LIST_ITEM* item=(_PROPERTY_LIST_ITEM*)pNMListView->lParam;
	if(item)
	{
		OnDeleteItemData(pNMListView->iItem, item->data);

		delete item->prop;
		delete item;
	}
	*pResult = 0;
}

void CEditableListCtrl::DrawInsert(int nIndex)
{
	LVINSERTMARK lvim;
	memset(&lvim, 0, sizeof(LVINSERTMARK));
	lvim.cbSize=sizeof(LVINSERTMARK);

	if(nIndex<0)
	{
		nIndex=GetItemCount()-1;
		lvim.dwFlags=LVIM_AFTER;
	}
	lvim.iItem=nIndex;
	SetInsertMark(&lvim);
}

void CEditableListCtrl::ResetDrawInsert()
{
	LVINSERTMARK lvim;
	memset(&lvim, 0, sizeof(LVINSERTMARK));
	lvim.cbSize=sizeof(LVINSERTMARK);
	lvim.iItem=-1;
	SetInsertMark(&lvim);
}

void CEditableListCtrl::OnBegindrag(NMHDR* pNMHDR, LRESULT* pResult) 
{
	NM_LISTVIEW* pNMListView = (NM_LISTVIEW*)pNMHDR;
	// TODO: Add your control notification handler code here

	m_nLast = -1;

	if(m_pDropTarget)
	{
		TRACE(_T("list DoDragDrop\r\n"));
		CSharedFile sf;

		_LISTCTRL_DRAG_CLIPBOARD dragInfo;
		dragInfo.pPointer=this;
		dragInfo.iDragItem=pNMListView->iItem;

		sf.Write(&dragInfo, sizeof(dragInfo));
		HGLOBAL hMem=sf.Detach();
		if(hMem)
		{
			COleDataSource oleSourceObj;
			oleSourceObj.CacheGlobalData(m_cf, hMem);
			oleSourceObj.DoDragDrop(DROPEFFECT_LINK);
		}

		TRACE(_T("list end drag\r\n"));
	}
	*pResult = 0;
}

void CEditableListCtrl::SelectItem(int iItem)
{
	// 선택항목 select 
	SetItemState(iItem, LVIS_SELECTED|LVIS_FOCUSED, LVIS_SELECTED|LVIS_FOCUSED);
	int nBefore = SetSelectionMark(iItem);

	Update(iItem);

	SetFocus();
}

void CEditableListCtrl::OnNcLButtonDown(UINT nHitTest, CPoint point)
{
	// 포커스를 확실히 가져와야, 리스트내에 보이는 edit box 와 combo box 를 숨길 수 있다.
	// 즉, OnKillfocusCmbBox 와 OnKillfocusEditBox 가 호출 될 수 있다.
	SetFocus();
	__super::OnNcLButtonDown(nHitTest, point);
}


void CEditableListCtrl::OnLvnInsertitem(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	ResizeHeader();
}

void CEditableListCtrl::ResizeHeader()
{
	CHeaderCtrl* pHeaderCtrl = GetHeaderCtrl();
	int nColumn=pHeaderCtrl->GetItemCount();
	for(int i=0;i<nColumn;i++)
	{
		SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);
	}
}

void CEditableListCtrl::OnSize(UINT nType, int cx, int cy)
{
	__super::OnSize(nType, cx, cy);

	ResizeHeader();
}

int CEditableListCtrl::GetSelectedItem()
{
	POSITION pos=GetFirstSelectedItemPosition();
	if(pos==NULL)
		return -1;

	return GetNextSelectedItem(pos);
}

void CEditableListCtrl::SelectedItemMoveUp()
{
	int selected_target=GetSelectedItem();
	if(selected_target<1)
		return;

	SetFocus();
	MoveItem(selected_target, selected_target-1);
}

void CEditableListCtrl::SelectedItemMoveTop()
{
	int selected_target=GetSelectedItem();
	if(selected_target<1)
		return;

	SetFocus();
	MoveItem(selected_target, 0);
}

void CEditableListCtrl::SelectedItemMoveDown()
{
	int selected_target=GetSelectedItem();
	if(selected_target<0)
		return;

	SetFocus();

	int n=GetItemCount();
	if(selected_target>=n-1)
		return;

	MoveItem(selected_target, selected_target+2);
}

void CEditableListCtrl::SelectedItemMoveBottom()
{
	int selected_target=GetSelectedItem();
	if(selected_target<0)
		return;

	SetFocus();

	int n=GetItemCount();
	if(selected_target>=n-1)
		return;

	MoveItem(selected_target, n);
}

void CEditableListCtrl::SelectedItemDelete()
{
	int selected_target=GetSelectedItem();
	if(selected_target<0)
		return;

	if(GetItemCount()==1)
		DeleteAllItems();
	else
	{
		DeleteItem(selected_target);
		if(selected_target<GetItemCount())
			SelectItem(selected_target);

		SetFocus();
	}
}
