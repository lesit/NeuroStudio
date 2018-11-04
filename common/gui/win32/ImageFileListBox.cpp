#include "stdafx.h"

#include "ImageFileListBox.h"

using namespace ahnn::windows;

CImageFileListBox::CImageFileListBox()
{
	m_cxImgSize.cx=120;
	m_cxImgSize.cy=100;
}

CImageFileListBox::~CImageFileListBox()
{
}

BEGIN_MESSAGE_MAP(CImageFileListBox, CImageListBox)
	ON_WM_CREATE()
END_MESSAGE_MAP()

int CImageFileListBox::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	if(m_imgList.GetSafeHandle()==NULL)
		m_imgList.Create(m_cxImgSize.cx, m_cxImgSize.cy, ILC_COLOR24, 0, 1);

	SetImageList(&m_imgList);

	return 0;
}

bool CImageFileListBox::AddFile(const wchar_t* file_path)
{
	Gdiplus::Bitmap orgImg(file_path);
	if(orgImg.GetLastStatus() != Gdiplus::Ok )
		return false;
	
	int nItem=GetCount();

	m_imgList.SetImageCount(nItem+1);
	{
		NP_2DSHAPE rect=np::GetRatioShape(NP_SIZE(orgImg.GetWidth(), orgImg.GetHeight()), NP_2DSHAPE(m_cxImgSize.cx, m_cxImgSize.cy));

		Gdiplus::Bitmap* pThumbnail = static_cast<Gdiplus::Bitmap*>(orgImg.GetThumbnailImage(rect.sz.width, rect.sz.height, NULL, NULL));
		
		// attach the thumbnail bitmap handle to an CBitmap object
		HBITMAP	hBmp = NULL;
		pThumbnail->GetHBITMAP(NULL, &hBmp);
		CBitmap image;
		image.Attach(hBmp);
		m_imgList.Replace(nItem, &image, NULL);

		delete pThumbnail;
	}

	int iInsert=AddString(JNM_Util::GetFileName(file_path), nItem);
	EnableItem(iInsert);
	SetItemDataPtr(iInsert, new std::wstring(file_path));

//	Invalidate();
	return true;
}

void CImageFileListBox::DeleteItemData(int nIndex)
{
	std::wstring* path=(std::wstring*)GetItemDataPtr(nIndex);
	if(path)
		delete path;

	__super::DeleteItemData(nIndex);
}
