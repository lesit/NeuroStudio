#if !defined(_GRAPH_DATA_SOURCE_ABSTRACT_H)
#define _GRAPH_DATA_SOURCE_ABSTRACT_H

#include "../common.h"

namespace np
{
	namespace gui
	{
		struct _graphLabel
		{
			_graphLabel()
			{
				memset(this, 0, sizeof(_graphLabel));
			}
			_graphLabel(neuron_value value, const wchar_t* label)
			{
				this->value=value;
				wcsncpy_s(this->label, label, _countof(this->label));
				this->label[_countof(this->label)-1]=0;
			}

			_graphLabel(neuron_value value)
			{
				this->value = value;
				swprintf(this->label, L"%f", value);
			}

			_graphLabel(neuro_32 value)
			{
				this->value = neuron_value(value);
				swprintf(this->label, L"%d", value);
			}
			_graphLabel(neuro_u32 value)
			{
				this->value = neuron_value(value);
				swprintf(this->label, L"%u", value);
			}

			_graphLabel(neuro_64 value)
			{
				this->value = neuron_value(value);
				swprintf(this->label, L"%lld", value);
			}
			_graphLabel(neuro_u64 value)
			{
				this->value = neuron_value(value);
				swprintf(this->label, L"%llu", value);
			}

			neuron_value value;
			wchar_t label[100];
		};

		struct _graph_line
		{
			_graph_line()
			{
				clr=RGB(0, 0, 0);
			}

			_graph_line(const _graph_line& src)
			{
				*this=src;
			}

			_graph_line& operator=(const _graph_line& src)
			{
				clr=src.clr;
				valueArray.assign(src.valueArray.begin(), src.valueArray.end());
				return *this;
			}

			COLORREF clr;
			_std_value_vector valueArray;
		};

		struct _graph_view
		{
			_graph_view()
			{
				upper_boundary=lower_boundary = 0.0;

				heightRatio=1.0;

				shapeType = _shape_type::dot;
			}
			_graph_view(const _graph_view& src)
			{
				*this=src;
			}

			_graph_view& operator=(const _graph_view& src)
			{
				graphLineVector = src.graphLineVector;
				yLabelArray = src.yLabelArray;

				curpos_yLabel_vector = src.curpos_yLabel_vector;

				lower_boundary = src.lower_boundary;
				upper_boundary = src.upper_boundary;

				heightRatio=src.heightRatio;

				shapeType = src.shapeType;
				return *this;
			}

			std::vector<_graph_line> graphLineVector;
			std::vector<_graphLabel> yLabelArray;
			
			std::vector<_graphLabel> curpos_yLabel_vector;

			neuron_value upper_boundary, lower_boundary;

			neuro_float heightRatio;

			enum class _shape_type{ dot, line, bar };
			_shape_type shapeType;
		};

		struct _graph_frame
		{
			_graph_frame()
			{
				nSkipData=0;
				has_cur_pos = false;
			}

			_graph_frame& operator=(const _graph_frame& src)
			{
				nSkipData = src.nSkipData;
				graphViewVector = src.graphViewVector;
				xLabelVector = src.xLabelVector;
				curpos_xLabel = src.curpos_xLabel;
			}

			neuro_u32 nSkipData;
			std::vector<_graph_view> graphViewVector;
			std::vector<_graphLabel> xLabelVector;

			bool has_cur_pos;
			_graphLabel curpos_xLabel;
		};

		class GraphDataSourceAbstract
		{
		public:
			GraphDataSourceAbstract(){}
			virtual ~GraphDataSourceAbstract(){}

			virtual neuro_u32 GetTotalScrollDataCount() const = 0;

			virtual bool IsValid(neuro_64 nStart, neuro_64 nCount) = 0;

			virtual bool GetViewData(neuro_64 nStart, neuro_64 nCount, neuro_u32 max_ylabel, neuro_64 cur_data_pos, _graph_frame& graphFrame) = 0;

			virtual CString GetDataTooltipLabel(neuro_u32 iGraph, neuro_u32 i) const {return _T("");}
		};
	}
}

#endif
