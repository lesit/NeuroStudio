#pragma once

#include "common.h"
#include "Windows/shape.h"

// CImageFileListCtrl
namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CImageFileListCtrl : public CListCtrl
			{
				DECLARE_DYNAMIC(CImageFileListCtrl)

			public:
				CImageFileListCtrl(NP_SIZE scale_size = NP_SIZE(80, 100), bool bAlignTop = true, bool bDrawFrameRect = true);
				virtual ~CImageFileListCtrl();

				bool Create(bool bThumbnail, DWORD dwStyle, const RECT& rect, CWnd* pParent, UINT nID);

				void ChangeListType(bool bThumbnail);

				void SetDrawFrameRect(bool bDrawFrameRect)
				{
					m_bDrawFrameRect = bDrawFrameRect;
				}

				void AddItemImagePath(const wchar_t* path);
				void SetItemImagePath(int index, const wchar_t* path);
				void DeleteItems(const std::vector<int>& item_vector);

				void MoveSelectedItem(bool bMoveUp);
				int MoveItem(int iItem, int iTarget);

				void GetFilePathVector(std_wstring_vector& vector) const;
			protected:
				DWORD GetListCtrlStyle(DWORD dwOldStyle);

				void RepositionItems(int iStart);

				int GetImageIndex(const wchar_t* path);

			protected:
				virtual void PreSubclassWindow();

				DECLARE_MESSAGE_MAP()
				afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
				afx_msg void OnSize(UINT nType, int cx, int cy);
				afx_msg void OnLvnDeleteitem(NMHDR *pNMHDR, LRESULT *pResult);

			private:
				CImageList m_imgList;
				CMapStringToPtr m_imageMap;

				bool m_bImage;
				bool m_bDrawFrameRect;

				NP_SIZE m_scale_size;
				bool m_bAlignTop;
			};
		}
	}
}
