#if !defined(AFX_EXTLISTCTRL_H__60770F23_A193_11D5_8D17_0000B4A89936__INCLUDED_)
#define AFX_EXTLISTCTRL_H__60770F23_A193_11D5_8D17_0000B4A89936__INCLUDED_

#include <afxtempl.h>
#include "afxole.h"

#pragma once
// ExtListCtrl.h : header file
//

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CEditableListCtrl;

			static const TCHAR* szListCtrlDragCF = _T("EditableList.Drag");
			struct _LISTCTRL_DRAG_CLIPBOARD
			{
				CEditableListCtrl* pPointer;
				int iDragItem;
			};

			/////////////////////////////////////////////////////////////////////////////
			// CExtListCtrl window
			class CEditableListCtrl : public CListCtrl
			{
			public:
				virtual const TCHAR* GetDragCF(){ return szListCtrlDragCF; }

				enum PROPERTYITEM_TYPE{ PIT_STATIC = 0, PIT_COMBO, PIT_EDIT, PIT_COLOR, PIT_FONT, PIT_FILE, PIT_BUTTON };
				struct CProperty
				{
					CProperty()
					{
						type = PIT_EDIT;
					}
					CProperty(const CProperty& src)
					{
						*this = src;
					}
					CProperty& operator=(const CProperty& src)
					{
						type = src.type;
						comboItemArray.Copy(src.comboItemArray);
						return *this;
					}
					PROPERTYITEM_TYPE type;

					CStringArray comboItemArray;
				};

				class CColumnProperty : public CArray<CProperty>
				{
				public:
					CColumnProperty();
					CColumnProperty(PROPERTYITEM_TYPE oneType);
					CColumnProperty(const CStringArray& oneComboItemArray);

					virtual ~CColumnProperty();
				public:
					void AddProperty(PROPERTYITEM_TYPE type);
					void AddProperty(const CStringArray& comboItemArray);

					const CProperty* GetProperty(int column) const;
				};

				struct _PROPERTY_LIST_ITEM
				{
					CColumnProperty* prop;
					DWORD_PTR data;
				};

				class CStructProperty{
				public:
					CStructProperty(){ m_iItem = -1; m_iSubItem = -1; };
					CComboBox m_cmbBox;
					CEdit m_editBox;
					CButton m_btnCtrl;

					int m_iItem;
					int m_iSubItem;
				};

				class CListDropTarget : public COleDropTarget
				{
				public:
					CListDropTarget(CEditableListCtrl& listCtrl);
					virtual ~CListDropTarget(){};

				protected:
					virtual DROPEFFECT OnDragEnter(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point);
					virtual DROPEFFECT OnDragOver(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point);
					virtual BOOL OnDrop(CWnd* pWnd, COleDataObject* pDataObject, DROPEFFECT dropEffect, CPoint point);
					virtual void OnDragLeave(CWnd* pWnd);

					bool GetSelfDrag(CWnd* pWnd, COleDataObject* pDataObject, _LISTCTRL_DRAG_CLIPBOARD& dragInfo);

					UINT m_cf;
				};

				// Construction
			public:
				CEditableListCtrl();
				virtual ~CEditableListCtrl();

				// Operations
			public:
				DECLARE_DYNAMIC(CEditableListCtrl)

				// Overrides
				// ClassWizard generated virtual function overrides
				//{{AFX_VIRTUAL(CExtListCtrl)
			protected:
				virtual void PreSubclassWindow();
				//}}AFX_VIRTUAL

				// Implementation
			public:
				void AddDefaultProperty(PROPERTYITEM_TYPE type);
				void AddDefaultProperty(const CStringArray& comboItemArray);
				void ResetDefaultProperty(){ m_defaultProperty.RemoveAll(); }
				CColumnProperty& GetDefaultProperty(){ return m_defaultProperty; }

				int InsertPropertyItem(int iInsert, LPCTSTR strLabel, CColumnProperty* pPropItem, DWORD_PTR data);
				int InsertItem(int iInsert, LPCTSTR strLabel, DWORD_PTR data = NULL);
				bool SetItemData(int iItem, DWORD_PTR data);
				DWORD_PTR GetItemData(int iItem) const;

				virtual const CProperty* GetProperty(int iItem, int iSubItem) const;

				void EditItem(int iItem, int iSubItem=0);

				virtual void MoveItem(int iItem, int iTarget);
				int GetSelectedItem();

				void DrawInsert(int nIndex);
				void ResetDrawInsert();
				void ResizeHeader();

				void SelectItem(int iItem);

				void SelectedItemMoveUp();
				void SelectedItemMoveTop();
				void SelectedItemMoveDown();
				void SelectedItemMoveBottom();
				void SelectedItemDelete();

				UINT GetClipboardFormat() const { return m_cf; }

			protected:
				virtual bool OnChangingProp(int iItem, int iSubItem, LPCTSTR newStr){ return true; }
				virtual bool OnChangedProp(int iItem, int iSubItem, LPCTSTR newStr);

				virtual bool OnUserButton(int iItem, int iSubItem){ return false; }

				virtual void OnDeleteItemData(int iItem, DWORD_PTR data){};
				// Generated message map functions
			protected:
				//{{AFX_MSG(CExtListCtrl)
				afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
				afx_msg void OnDestroy();
				afx_msg void OnNMClick(NMHDR* pNMHDR, LRESULT* pResult);
				afx_msg void OnBegindrag(NMHDR* pNMHDR, LRESULT* pResult);
				afx_msg void OnDeleteitem(NMHDR* pNMHDR, LRESULT* pResult);
				//}}AFX_MSG
				afx_msg void OnKillfocusCmbBox();
				afx_msg void OnSelchangeCmbBox();
				afx_msg void OnKillfocusEditBox();
				afx_msg void OnButton();
				afx_msg void OnNcLButtonDown(UINT nHitTest, CPoint point);
				afx_msg void OnLvnInsertitem(NMHDR *pNMHDR, LRESULT *pResult);
				afx_msg void OnSize(UINT nType, int cx, int cy);
				DECLARE_MESSAGE_MAP()

			protected:
				virtual CListDropTarget* CreateDropTarget(){ return new CListDropTarget(*this); }
				UINT m_cf;

			private:
				CListDropTarget* m_pDropTarget;

				CStructProperty m_stProp;

				CColumnProperty m_defaultProperty;

				int m_nLast;
			};
		}
	}
}
/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_EXTLISTCTRL_H__60770F23_A193_11D5_8D17_0000B4A89936__INCLUDED_)
