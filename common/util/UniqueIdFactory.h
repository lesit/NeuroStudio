#if !defined(_UNIQUE_ID_FACTORY_H)
#define _UNIQUE_ID_FACTORY_H

#include <set>

#include "../common.h"

namespace np
{
	namespace util
	{
		class UniqueIdFactory
		{
		private:
			const neuro_u32 start_id;
			const neuro_u32 last_id;

			std::set<neuro_u32> m_id_set;
		public:
			UniqueIdFactory(neuro_u32 _start_id = 0, neuro_u32 _last_id = neuro_last32)
				: start_id(_start_id), last_id(_start_id < _last_id ? _last_id : neuro_last32)
			{
			}
			
			UniqueIdFactory& operator = (const UniqueIdFactory& src)
			{
				const_cast<neuro_u32&>(start_id) = src.start_id;
				const_cast<neuro_u32&>(last_id) = src.last_id;

				m_id_set = src.m_id_set;
				return *this;
			}

			neuro_u32 CreateId()
			{
				if (m_id_set.size() == 0)
				{
					m_id_set.insert(start_id);
					return start_id;
				}

				if (m_id_set.size() == last_id - start_id)
					return neuro_last32;

				neuro_u32 new_id = *m_id_set.rbegin();
				if (new_id + 1 < last_id)	// 마지막 uid 후의 값을 할당할 수 있다면..
				{
					m_id_set.insert(m_id_set.end(), ++new_id);
					return new_id;
				}

				new_id = start_id;
				std::set<neuro_u32>::const_iterator it_end = m_id_set.end();
				for (std::set<neuro_u32>::const_iterator it = m_id_set.begin(); it != it_end; it++)
				{
					if (new_id < *it)	// 비어 있는 것을 찾았을 경우
					{
						m_id_set.insert(it, new_id);
						return start_id + new_id;
					}

					new_id = *it + 1;
				}
				return neuro_last32;
			}

			bool InsertId(neuro_u32 id)
			{
				if(id < start_id || id >= last_id)
					return false;

				if (m_id_set.find(id) != m_id_set.end())
					return false;

				m_id_set.insert(id);
				return true;
			}

			void RemoveId(neuro_u32 id)
			{
				if (m_id_set.size() == 0)
					return;

				std::set<neuro_u32>::iterator it = m_id_set.find(id);
				if (it != m_id_set.end())
					m_id_set.erase(it);
			}

			bool HasId(neuro_u32 id)
			{
				return m_id_set.find(id) != m_id_set.end();
			}

			void RemoveAll()
			{
				m_id_set.clear();
			}
		};
	}
}

#endif
