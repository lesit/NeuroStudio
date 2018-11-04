// ThumbnailListCtrl.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "ImageFileListCtrl.h"
#include "ImageUtil.h"

// CImageFileListCtrl
using namespace np::gui::win32;

IMPLEMENT_DYNAMIC(CImageFileListCtrl, CListCtrl)

CImageFileListCtrl::CImageFileListCtrl(NP_SIZE scale_size, bool bAlignTop, bool bDrawFrameRect)
{
	m_scale_size=scale_size;

	m_bImage=false;
	m_bDrawFrameRect=bDrawFrameRect;

	m_bAlignTop=bAlignTop;
}

CImageFileListCtrl::~CImageFileListCtrl()
{
}

BEGIN_MESSAGE_MAP(CImageFileListCtrl, CListCtrl)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_NOTIFY_REFLECT(LVN_DELETEITEM, &CImageFileListCtrl::OnLvnDeleteitem)
END_MESSAGE_MAP()

DWORD CImageFileListCtrl::GetListCtrlStyle(DWORD dwOldStyle)
{
	DWORD dwStyle = dwOldStyle | WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_SHOWSELALWAYS;

	dwStyle&=~LVS_ALIGNLEFT;
	dwStyle&=~LVS_ALIGNTOP;
	if(m_bImage)
	{
		dwStyle&=~LVS_LIST;

		if(m_bAlignTop)
			dwStyle|=LVS_ALIGNTOP;
		else
			dwStyle|=LVS_ALIGNLEFT;
		dwStyle|=LVS_ICON;
	}
	else
	{
		dwStyle&=~LVS_ICON;
		dwStyle|=LVS_LIST;
	}
	return dwStyle;
}

bool CImageFileListCtrl::Create(bool bThumbnail, DWORD dwStyle, const RECT& rect, CWnd* pParent, UINT nID)
{
	m_bImage=bThumbnail;

	return __super::Create(GetListCtrlStyle(dwStyle), rect, pParent, nID)!=FALSE;
}

int CImageFileListCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CListCtrl::OnCreate(lpCreateStruct) == -1)
		return -1;

	if(m_imgList.GetSafeHandle()==NULL)
		m_imgList.Create(m_scale_size.width, m_scale_size.height, ILC_COLOR24, 0, 1);

	SetImageList(&m_imgList, LVSIL_NORMAL);
	return 0;
}

void CImageFileListCtrl::PreSubclassWindow()
{
	CListCtrl::PreSubclassWindow();

	// 흠.. PreSubclassWindow 부터 호출되고 OnCreate 가 호출된다.
	/*	그리고, 아래 SetWindowLong이 먹지를 않는다 ㅠㅠ
	DWORD dwFlags = WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_SHOWSELALWAYS|LVS_ICON|LVS_SINGLESEL;//|LVS_AUTOARRANGE;
	SetWindowLong(m_hWnd, GWL_STYLE, dwFlags);
	*/

	m_imgList.Create(m_scale_size.width, m_scale_size.height, ILC_COLOR24, 0, 1);
	SetImageList(&m_imgList, LVSIL_NORMAL);
}

void CImageFileListCtrl::ChangeListType(bool bThumbnail)
{
	DeleteAllItems();

	DWORD dwNewStyle=GetListCtrlStyle(GetWindowLong(m_hWnd, GWL_STYLE));
	SetWindowLong(m_hWnd, GWL_STYLE, dwNewStyle);

	m_bImage=bThumbnail;
}

// CImageFileListCtrl 메시지 처리기입니다.
void CImageFileListCtrl::OnSize(UINT nType, int cx, int cy)
{
	CListCtrl::OnSize(nType, cx, cy);
}

int CImageFileListCtrl::GetImageIndex(const wchar_t* path)
{
	CString strLowcasePath(path);
	strLowcasePath.MakeLower();

	void* index;
	if(m_imageMap.Lookup(strLowcasePath, index))
		return (int)index;

	Gdiplus::Bitmap orgImg(path);
	if(orgImg.GetLastStatus() != Gdiplus::Ok )
		return -1;
	
	int iImage=m_imgList.GetImageCount();
	m_imgList.SetImageCount(iImage+1);
	
	util::ReadImage image(3, m_scale_size.width, m_scale_size.height);
	if(!image.LoadImage(path))
		return -1;

	ImageList_Replace(m_imgList.GetSafeHandle(), iImage, image.GetImage(), NULL);
/*
	{
		NP_2DSHAPE rect=NP_Util::GetRatioShape(NP_SIZE(orgImg.GetWidth(), orgImg.GetHeight()), NP_2DSHAPE(m_scale_size.width, m_scale_size.height));

		Gdiplus::Bitmap* pThumbnail = static_cast<Gdiplus::Bitmap*>(orgImg.GetThumbnailImage(rect.sz.width, rect.sz.height, NULL, NULL));
		
		// attach the thumbnail bitmap handle to an CBitmap object
		HBITMAP	hBmp = NULL;
		pThumbnail->GetHBITMAP(NULL, &hBmp);
		CBitmap image;
		image.Attach(hBmp);
		m_imgList.Replace(iImage, &image, NULL);

		delete pThumbnail;
	}*/
	m_imageMap.SetAt(strLowcasePath, (void*)iImage);

	return iImage;
}

void CImageFileListCtrl::AddItemImagePath(const wchar_t* path)
{
	if(!m_bImage)
		return;

	int iImage=GetImageIndex(path);
	if(iImage<0)
		return;

	int nItem=GetItemCount();
	int iInsert=InsertItem(nItem,NP_Util::GetFileName(path), iImage);
	SetItemData(iInsert, (DWORD_PTR)new std::wstring(path));

	// get current item position
	if(m_bAlignTop && iInsert>0)
	{
		CRect rc;
		GetItemRect(iInsert-1, &rc, LVIR_BOUNDS);

		CPoint pt;
		GetItemPosition(iInsert-1, &pt);	 
	  
		// shift the thumbnail to desired position
		pt.y = iInsert*(rc.Height() + 20);
		SetItemPosition(iInsert, pt);
	}

	Invalidate();
}

void CImageFileListCtrl::SetItemImagePath(int index, const wchar_t* path)
{
	if(!m_bImage)
		return;

	if(index>=GetItemCount())
	{
		AddItemImagePath(path);
		return;
	}

	int iImage=GetImageIndex(path);
	if(iImage<0)
		return;

	std::wstring* ptr_path=(std::wstring*)GetItemData(index);
	ptr_path->assign(path);

	SetItem(index, 0, LVIF_TEXT|LVIF_IMAGE, NP_Util::GetFileName(path), iImage, 0, 0, 0);
}

void CImageFileListCtrl::DeleteItems(const std::vector<int>& item_vector)
{
	std::vector<int> sorted(item_vector);
	np::sort<int>(sorted, true);

	TRACE(L"CImageFileListCtrl::DeleteFiles : start\r\n");

	std::vector<int>::const_iterator it=sorted.begin();
	for(;it!=sorted.end();it++)
	{
		int i=*it;
		TRACE(L"%d, ", i);
		DeleteItem(*it);
	}

	RepositionItems(0);
}

void CImageFileListCtrl::RepositionItems(int iStart)
{
	if(!m_bImage || !m_bAlignTop)
		return;

	int n=GetItemCount();
	if(n==0)
		return;

	// reposition of items
	CRect rc;
	GetItemRect(0, &rc, LVIR_BOUNDS);

	for(int i=0;i<n;i++)
	{
		CPoint pt;
		GetItemPosition(i, &pt);	 
	  
		// shift the thumbnail to desired position
		pt.x = 10;
		pt.y = i*(rc.Height() + 20);

		SetItemPosition(i, pt);
	}

	TRACE(L"CImageFileListCtrl::DeleteFiles : end\r\n");
}

void CImageFileListCtrl::MoveSelectedItem(bool bMoveUp)
{
	POSITION pos = GetFirstSelectedItemPosition();
	if(!pos)
		return;
	int iItem = GetNextSelectedItem(pos);

	while(pos)	// 일단 나머지는 선택 해제시킨다.
	{
		int other = GetNextSelectedItem(pos);

		SetItemState(other, LVIS_SELECTED|LVIS_FOCUSED, LVIS_SELECTED|LVIS_FOCUSED);
		Update(other);
	}
	SetFocus();

	int iTarget=bMoveUp ? (iItem-1) : (iItem+2);
	if(iTarget<0 || iTarget>=GetItemCount()+1)
		return;

	iTarget=MoveItem(iItem, iTarget);
	if(iTarget<0)
		return;

	SetItemState(iTarget, LVIS_SELECTED|LVIS_FOCUSED, LVIS_SELECTED|LVIS_FOCUSED);
	int nBefore = SetSelectionMark(iTarget);

	Update(iTarget);
}

int CImageFileListCtrl::MoveItem(int iItem, int iTarget)
{
	if(iTarget<0)
		iTarget=GetItemCount();

	if (iItem < iTarget)	// 만약에 방금 삭제한 곳이 이동하려는 곳보다 먼저이면
		--iTarget;

	if(iItem==iTarget)
		return -1;

	CStringArray strBufArray;
	CArray<LPCTSTR> labelArray;
	if(GetHeaderCtrl())
	{
		int nText=GetHeaderCtrl()->GetItemCount();
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
	}
	else
	{
		strBufArray.Add(GetItemText(iItem, 0));
		labelArray.Add(strBufArray[0]);
	}

	LVITEM item;
	memset(&item, 0, sizeof(LVITEM));
	item.mask=LVIF_IMAGE|LVIF_PARAM;
	item.iItem=iItem;
	item.iSubItem=0;
	GetItem(&item);

	__super::SetItemData(iItem, NULL);
	DeleteItem(iItem);

	iTarget=__super::InsertItem(iTarget, labelArray[0], item.iImage);
	__super::SetItemData(iTarget, item.lParam);
	for(int i=1;i<labelArray.GetSize();i++)
		SetItemText(iTarget, i, labelArray[i]);

	TRACE(L"move(%d, %d) : %X\r\n", iItem, iTarget, item.lParam);

	RepositionItems(min(iItem, iTarget));

	EnsureVisible(iTarget, FALSE);
	return iTarget;
}

void CImageFileListCtrl::OnLvnDeleteitem(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	if(!m_bImage)
		return;

	std::wstring* path=(std::wstring*)pNMLV->lParam;
	if(path)
		delete path;
}

void CImageFileListCtrl::GetFilePathVector(std_wstring_vector& vector) const
{
	if(!m_bImage)
		return;

	int n=GetItemCount();
	for(int i=0;i<n;i++)
	{
		std::wstring* path=(std::wstring*)GetItemData(i);
		if(path)
		{
			DEBUG_OUTPUT(path->c_str());
			vector.push_back(path->c_str());
		}
	}
}
