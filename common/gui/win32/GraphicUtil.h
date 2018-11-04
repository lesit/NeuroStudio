#if !defined(_GRAPHIC_UTIL_H)
#define _GRAPHIC_UTIL_H

#include "../line.h"

#include <atlimage.h>

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class GraphicUtil
			{
			public:
				inline static Gdiplus::Color Transform(COLORREF clr)
				{
					return Gdiplus::Color(GetRValue(clr), GetGValue(clr), GetBValue(clr));
				}

				static inline void DrawPoint(HDC dc, const NP_POINT& pt, const Gdiplus::Pen& pen)
				{
					Gdiplus::Graphics graphic(dc);
					graphic.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
					graphic.DrawEllipse(&pen, pt.x - 1, pt.y - 1, 2, 2);
				}

				static inline void DrawRect(HDC dc, const NP_RECT& rc, const Gdiplus::Pen& pen)
				{
					Gdiplus::Graphics graphic(dc);
					graphic.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
					graphic.DrawRectangle(&pen, rc.left, rc.top, rc.GetWidth(), rc.GetHeight());
				}

				static inline void CompositeLinePen(Gdiplus::Pen& pen, _line_arrow_type arrow_type, _line_dash_type dash_type)
				{
					pen.SetLineJoin(Gdiplus::LineJoinRound);

					Gdiplus::AdjustableArrowCap arrowcap(5, 5);
					arrowcap.SetMiddleInset(2);
					pen.SetCustomStartCap(arrow_type == _line_arrow_type::start || arrow_type == _line_arrow_type::both ? &arrowcap : NULL);
					pen.SetCustomEndCap(arrow_type == _line_arrow_type::end || arrow_type == _line_arrow_type::both ? &arrowcap : NULL);

					switch (dash_type)
					{
					case _line_dash_type::dot:
						pen.SetDashStyle(Gdiplus::DashStyleDot);
						break;
					case _line_dash_type::dash:
						pen.SetDashStyle(Gdiplus::DashStyleDash);
						break;
					case _line_dash_type::dash_dot:
						pen.SetDashStyle(Gdiplus::DashStyleDashDot);
						break;
					case _line_dash_type::dash_dot_dot:
						pen.SetDashStyle(Gdiplus::DashStyleDashDotDot);
						break;
					}
				}

				inline static void DrawLine(HDC dc, const NP_POINT& from, const NP_POINT& to, COLORREF clr, int line_width = 1, int arrow_width = 0, int arrow_inner_gap = 0)
				{
					Gdiplus::Pen pen(Transform(clr), line_width);
					pen.SetLineJoin(Gdiplus::LineJoinRound);
					if (arrow_inner_gap > 0 && arrow_inner_gap < arrow_width)
					{
						Gdiplus::AdjustableArrowCap cap(arrow_width, arrow_width, true);
						cap.SetMiddleInset(arrow_inner_gap);
						pen.SetCustomEndCap(&cap);
					}

					DrawLine(dc, from, to, pen);
				}

				inline static void DrawLine(HDC dc, const NP_POINT& from, const NP_POINT& to, const Gdiplus::Pen& pen)
				{
					Gdiplus::Graphics graphic(dc);
					graphic.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
					graphic.DrawLine(&pen, from.x, from.y, to.x, to.y);
				}

				inline static bool LineHitTest(const NP_POINT& pt, const NP_POINT& from, const NP_POINT& to, int line_width, int arrow_width = 5, int arrow_inner_gap = 2)
				{
					Gdiplus::Pen pen(Gdiplus::Color(0,0,0), line_width);
					pen.SetLineJoin(Gdiplus::LineJoinRound);
					if (arrow_inner_gap > 0 && arrow_inner_gap < arrow_width)
					{
						Gdiplus::AdjustableArrowCap cap(arrow_width, arrow_width, true);
						cap.SetMiddleInset(arrow_inner_gap);
						pen.SetCustomEndCap(&cap);
					}
					return LineHitTest(pt, from, to, pen);
				}

				inline static bool LineHitTest(const NP_POINT& pt, const NP_POINT& from, const NP_POINT& to, const Gdiplus::Pen& pen)
				{
					Gdiplus::GraphicsPath path;
					path.AddLine(from.x, from.y, to.x, to.y);

					return path.IsOutlineVisible(pt.x, pt.y, &pen) != false;
				}

				inline static void DrawBezierLine(HDC dc, const _BEZIER_LINE& line, const Gdiplus::Pen& pen)
				{
					if (line.points.size() < 1)
						return;

					Gdiplus::GraphicsPath path;
					SetBezierLinePath(line, path);

					Gdiplus::Graphics graphic(dc);
					graphic.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
					graphic.DrawPath(&pen, &path);
				}

				inline static bool BezierLineHitTest(const NP_POINT& pt, const _BEZIER_LINE& line, const Gdiplus::Pen& pen)
				{
					if (line.points.size() < 1)
						return false;

					Gdiplus::GraphicsPath path;
					SetBezierLinePath(line, path);

					return path.IsOutlineVisible(pt.x, pt.y, &pen) != false;
				}

				inline static void SetBezierLinePath(const _BEZIER_LINE& line, Gdiplus::GraphicsPath& path)
				{
					NP_POINT from = line.start;
					for (neuro_u32 i = 0; i < line.points.size(); i++)
					{
						const _BEZIER_POINT& to = line.points[i];
						neuro_32 x_axis = 0;
						neuro_32 y_axis = 0;
						if(to.is_upward)
							x_axis = to.tension * neuro_float(to.pt.x - from.x) / 2.f;
						else
							y_axis = to.tension * neuro_float(to.pt.y - from.y) / 2.f;

						neuro_u32 x1 = from.x + x_axis;
						neuro_u32 y1 = from.y + y_axis;

						neuro_u32 x2, y2;
						if (to.two_axis)
						{
							x2 = to.pt.x - x_axis;
							y2 = to.pt.y - y_axis;
						}
						else
						{
							x2 = to.pt.x;
							y2 = to.pt.y;
						}
						path.AddBezier(from.x, from.y, x1, y1, x2, y2, to.pt.x, to.pt.y);

						from = to.pt;
					}
				}

				inline static void DrawCurvedLine(HDC dc, const _np_point_vector& points, const Gdiplus::Pen& pen)
				{
					if (points.size() < 1)
						return;

					Gdiplus::GraphicsPath path;
					SetCurvedLinePath(points, path);

					Gdiplus::Graphics graphic(dc);
					graphic.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
					graphic.DrawPath(&pen, &path);
				}

				inline static bool CurvedLineHitTest(const NP_POINT& pt, const _np_point_vector& points, const Gdiplus::Pen& pen)
				{
					if (points.size() < 1)
						return false;

					Gdiplus::GraphicsPath path;
					SetCurvedLinePath(points, path);

					return path.IsOutlineVisible(pt.x, pt.y, &pen) != false;
				}

				inline static void SetCurvedLinePath(const _np_point_vector& points, Gdiplus::GraphicsPath& path)
				{
					int point_count = points.size();
					Gdiplus::Point* line_points = new Gdiplus::Point[point_count];
					for (size_t i = 0; i < point_count; i++)
					{
						line_points[i].X = points[i].x;
						line_points[i].Y = points[i].y;
					}
					path.AddCurve(line_points, point_count);
					delete[] line_points;
				}
			};
		}
	}
}
#endif
