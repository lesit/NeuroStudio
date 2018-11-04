#pragma once

#include "gui/shape.h"
namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CScrollWnd : public CWnd
			{
			public:
				CScrollWnd();
				virtual ~CScrollWnd();

				void RefreshDisplay(const NP_RECT* area = NULL);
				void RefreshScrollBars();

				NP_POINT GetViewport();
				void SetViewport(const NP_POINT& org);

				inline neuro_32 ViewportToWndX(neuro_32 x) const
				{
					return x - m_vpOrg.x;
				}
				inline neuro_32 ViewportToWndY(neuro_32 y) const
				{
					return y - m_vpOrg.y;
				}
				inline NP_POINT ViewportToWnd(const NP_POINT& vp_pt) const
				{
					return{ ViewportToWndX(vp_pt.x), ViewportToWndY(vp_pt.y) };
				}
				inline NP_RECT ViewportToWnd(const NP_RECT& vp_rc) const
				{
					return NP_RECT(ViewportToWnd({ vp_rc.left, vp_rc.top }), ViewportToWnd({ vp_rc.right, vp_rc.bottom }));
				}
				inline neuro_32 WndToViewportX(neuro_32 x) const
				{
					return x + m_vpOrg.x;
				}
				inline neuro_32 WndToViewportY(neuro_32 y) const
				{
					return y + m_vpOrg.y;
				}
				inline NP_POINT WndToViewport(const NP_POINT& wnd_pt) const
				{
					return{ WndToViewportX(wnd_pt.x), WndToViewportY(wnd_pt.y) };
				}
				inline NP_RECT WndToViewport(const NP_RECT& wnd_rc) const
				{
					return NP_RECT(WndToViewport({ wnd_rc.left, wnd_rc.top }), WndToViewport({ wnd_rc.right, wnd_rc.bottom }));
				};

			protected:
				virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
				DECLARE_MESSAGE_MAP()
				afx_msg void OnSize(UINT nType, int cx, int cy);
				afx_msg void OnHScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar);
				afx_msg void OnVScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar);

			protected:
				virtual NP_SIZE GetScrollTotalViewSize() const = 0;
				virtual neuro_u32 GetScrollMoving(bool is_horz) const { return 1; }
				virtual neuro_u32 GetScrollTrackPos(bool is_horz) const { return 1; }

				virtual void OnScrollChanged() {
					RefreshDisplay();
				}

				NP_POINT m_vpOrg;
			};
		}
	}
}
