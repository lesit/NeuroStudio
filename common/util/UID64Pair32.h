#if !defined(_UID_64_PAIR_32_H)
#define _UID_64_PAIR_32_H

#include "../np_types.h"

namespace np
{
	struct UID64Pair32
	{
	private:
		struct pair
		{
			neuro_u32 lower;
			neuro_u32 upper;
		};
		union
		{
			pair id;
			neuro_u64 uid64;
		};

	public:
		UID64Pair32()
		{
			Initialize();
		}

		UID64Pair32(neuro_u32 upper, neuro_u32 lower)
		{
			SetUID(upper, lower);
		}

		UID64Pair32(neuro_u64 _uid64)
		{
			uid64 = _uid64;
		}

		void Initialize(){ uid64 = invalid_uid64; }	

		operator neuro_u64 () const { return uid64; }

		neuro_u32 upper_uid() const { return id.upper; }
		neuro_u32 lower_uid() const { return id.lower; }
		neuro_u64 uion_uid() const { return uid64; }

		void SetUID(neuro_u32 upper, neuro_u32 lower)
		{
			id.upper = upper;
			id.lower = lower;
		}

		static neuro_u32 GetUpper(neuro_u64 _uid64)
		{
			return UID64Pair32(_uid64).id.upper;
		}

		static neuro_u32 GetLower(neuro_u64 _uid64)
		{
			return UID64Pair32(_uid64).id.lower;
		}
	};
}

#endif
