#if !defined(_U32_BITMAP_VECTOR_H)
#define _U32_BITMAP_VECTOR_H

#include "common.h"

namespace np
{
	namespace util
	{
		class U32Bitmap
		{
		public:
			U32Bitmap()
			{
				bitmap = NULL;
				m_bit_count = m_fill_count = 0;
			}

			virtual ~U32Bitmap()
			{
				if (bitmap != NULL)
					free(bitmap);
			}

			bool InitBitmap(neuro_u32 count, bool fill=true)
			{
				if (bitmap != NULL)
					return false;

				neuro_u64 array_count = JNM_Util::CalculateCountPer(count, sizeof(neuro_u32));
				if (fill)
				{
					bitmap = (neuro_u32*)malloc(array_count * sizeof(neuro_u32));
					FillBitmap(count);
				}
				else
				{
					bitmap = (neuro_u32*)calloc(array_count, sizeof(neuro_u32));
					m_fill_count = 0;
				}
				m_bit_count = count;
				return true;
			}

			#define U32BITMAP_BIT(bit) (1<<bit)
			void FillBitmap(neuro_u32 count)
			{
				neuro_u64 array_count = _msize(bitmap) / sizeof(neuro_u32);

				neuro_u64 fill_count = count / sizeof(neuro_u32);
				for (neuro_u64 i = 0; i < fill_count; i++)
					bitmap[i] = neuro_last32;

				if (array_count > fill_count)
				{
					bitmap[fill_count] = 0;

					neuro_u64 last_bit = count % sizeof(neuro_u32);
					for (neuro_u64 i = 0; i < last_bit; i++)
						bitmap[fill_count] |= U32BITMAP_BIT(i);

					for (neuro_u64 i = fill_count + 1; i < array_count; i++)
						bitmap[i] = 0;
				}

				m_fill_count = count;
			}

			#define U32BITMAP_IS_SET(array_index, bit) ((bitmap[array_index] & U32BITMAP_BIT(bit)) != 0)
			bool SetBitmap(neuro_u32 index, bool set)
			{
				if (bitmap == NULL || index > m_bit_count)
					return false;

				neuro_u32 array_index = index / sizeof(neuro_u32);
				neuro_u8 bit = index % sizeof(neuro_u32);
				if (set)
				{
					if (!U32BITMAP_IS_SET(array_index, bit))
					{
						++m_fill_count;
						bitmap[array_index] |= U32BITMAP_BIT(bit);
					}
				}
				else
				{
					if (m_fill_count == 0)
						return false;

					if (U32BITMAP_IS_SET(array_index, bit))
					{
						--m_fill_count;
						bitmap[array_index] &= ~U32BITMAP_BIT(bit);
					}
				}
				return true;
			}

			bool IsSet(neuro_u32 index) const
			{
				if (bitmap == NULL || index > m_bit_count)
					return false;

				neuro_u32 array_index = index / sizeof(neuro_u32);
				neuro_u8 bit = index % sizeof(neuro_u32);
				return U32BITMAP_IS_SET(array_index, bit);
			}

			neuro_u32 GetBitCount() const { return m_bit_count; }
			neuro_u32 GetFillCount() const { return m_fill_count; }

		protected:
			
		private:
			neuro_u32* bitmap;
			neuro_u32 m_bit_count;
			neuro_u32 m_fill_count;
		};
	}
}
#endif
