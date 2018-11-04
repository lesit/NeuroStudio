#pragma once

namespace ahnn
{
	namespace windows
	{
		class CDragPropertyCtrl : public CMFCPropertyGridCtrl
		{
		public:
			CDragPropertyCtrl();
			virtual ~CDragPropertyCtrl();

			// Override to respond to beginning of drag event.
			virtual BOOL BeginDrag(_In_ CPoint pt);

			// Overrdie to react to user cancelling drag.
			virtual void CancelDrag(_In_ CPoint pt);

			// Called as user drags. Return constant indicating cursor.
			virtual UINT Dragging(_In_ CPoint pt);

			// Called when user releases mouse button to end drag event.
			virtual void Dropped(_In_ CPoint pt);

			const CList<CMFCPropertyGridProperty*, CMFCPropertyGridProperty*>& GetPropList() const {return m_lstProps;}

		protected:
			virtual void PreSubclassWindow();

			virtual BOOL OnChildNotify(UINT, WPARAM, LPARAM, LRESULT*);
			DECLARE_MESSAGE_MAP()

		protected:
			// draws insertion line
			virtual void DrawInsert(_In_ CMFCPropertyGridProperty* pProp);
			void DrawSingle(_In_ CMFCPropertyGridProperty* pProp);

			CMFCPropertyGridProperty* m_pLastProp;
		};
	}
}
