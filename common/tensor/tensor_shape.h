#if !defined(_TENSOR_SHAPE_H)
#define _TENSOR_SHAPE_H

#include <vector>

#include "np_types.h"

namespace np
{
	namespace tensor
	{
		class DataShape : public _std_u32_vector
		{
		public:
			DataShape()
			{
			}

			DataShape(const _std_u32_vector& src)
			{
				*this = src;
			}

			DataShape(const DataShape& src)
				:DataShape((const _std_u32_vector&)src)
			{
			}

			virtual ~DataShape()
			{}


			DataShape& operator = (const DataShape& src)
			{
				*this = (const _std_u32_vector&)src;

				return *this;
			}

			DataShape& operator = (const _std_u32_vector& src)
			{
				resize(src.size());
				for (size_t i = 0, n = src.size(); i < n; i++)
					at(i) = src[i]>1 ? src[i] : 1;

				return *this;
			}

			neuro_u32 GetChannelCount(neuro_u32 max_dim = 1) const
			{
				return GetDimSize(0, max_dim);
			}

			neuro_u32 GetHeight(neuro_u32 max_dim = 2) const
			{
				return GetDimSize(1, max_dim);
			}

			neuro_u32 GetWidth(neuro_u32 max_dim = neuro_last32) const
			{
				return GetDimSize(2, max_dim);
			}

			neuro_u32 GetDimSize(neuro_u32 start_dim = 0, neuro_u32 max_dim = neuro_last32) const
			{
				if (size() == 0)
					return 0;

				if (max_dim > size())
					max_dim = neuro_u32(size());

				neuro_u32 ret = 1;
				for (neuro_u32 i = start_dim; i < max_dim; i++)
					ret *= at(i);

				return ret;
			}

			bool IsEqual(const DataShape& target) const
			{
				if (size() != target.size())
					return false;

				for (size_t i = 0; i < size(); i++)
				{
					if (at(i) != target[i])
						return false;
				}
				return true;
			}
		};

		typedef std::vector<DataShape> _shape_vector;

		class TensorShape : public DataShape
		{
		public:
			TensorShape()
			{
				time_length = 1;
			}

			TensorShape(const _std_u32_vector& src)
				: DataShape(src)
			{
				time_length = 1;
			}

			TensorShape(neuro_u32 time, neuro_u32 channel, neuro_u32 height, neuro_u32 width)
				: DataShape({ channel, height, width })
			{
				time_length = time;
			}

			TensorShape& operator = (const TensorShape& src)
			{
				(DataShape&)*this = src;
				time_length = src.time_length;
				return *this;
			}

			TensorShape& operator = (const DataShape& src)
			{
				(DataShape&)*this = src;

				return *this;
			}

			neuro_32 FindMatchAxisCount(const TensorShape& other, neuro_32 until) const
			{
				if (time_length != other.time_length)
					return -1;

				neuro_32 n = until;
				if (n > size())
					n = (neuro_32)size();
				if (n > other.size())
					n = (neuro_32)other.size();

				for (int axis = 0; axis < n; axis++)
				{
					if (other[axis] != at(axis))
						return axis;
				}
				return n;
			}

			neuro_u32 GetTensorSize() const
			{
				return time_length * GetDimSize();
			}

			bool IsEqual(const TensorShape& target) const
			{
				if (!__super::IsEqual(target))
					return false;

				if (time_length != target.time_length)
					return false;

				return true;
			}
			neuro_u32 time_length;
		};
	}
}
#endif
