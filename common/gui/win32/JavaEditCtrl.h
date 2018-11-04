#if !defined(_JAVA_EDIT_CTRL_H)
#define _JAVA_EDIT_CTRL_H

#include <afxrich.h>

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CJavaEditCtrl : public CRichEditCtrl
			{
				// Construction
			public:
				CJavaEditCtrl();

				/// header¸¸ ¹Ù²Û´Ù.
				void SetHeaderContent(CString strHeader);
				void SetBodyContent(LPCTSTR strBody);

				CString GetBody();
				// Operations
			public:
				// Overrides
				// ClassWizard generated virtual function overrides
				//{{AFX_VIRTUAL(CJavaEditCtrl)
			public:
				virtual BOOL PreTranslateMessage(MSG* pMsg);
			protected:
				virtual void PreSubclassWindow();
				//}}AFX_VIRTUAL

				// Implementation
			public:
				virtual ~CJavaEditCtrl();

				// Generated message map functions
			protected:
				//{{AFX_MSG(CJavaEditCtrl)
				afx_msg void OnEnKillfocus();
				afx_msg void OnEnChange();
				afx_msg void OnDestroy();
				//}}AFX_MSG

				DECLARE_MESSAGE_MAP()

			protected:
				bool GetEditableArea(long &nSt, long &nEd);
			};
		}
	}
}

#endif
