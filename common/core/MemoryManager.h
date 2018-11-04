#if !defined(_MEMORY_MANAGER_H)
#define _MEMORY_MANAGER_H

#include "../np_types.h"
#include <stdlib.h>
#include <memory.h>

#include "math_device.h"

namespace np
{
	namespace core
	{
		class MemoryManager
		{
		public:
			static MemoryManager& GetManager(math_device_type type);

			virtual math_device_type GetType() const = 0;

			virtual bool GetMemoryInfo(neuro_size_t& free, neuro_size_t& total) const = 0;

			virtual void* Alloc(void* old, neuro_size_t old_size, neuro_size_t new_size) const = 0;
			virtual bool Dealloc(void* old) const = 0;

			virtual bool SetZero(void* data, neuro_size_t size) const = 0;

			virtual bool DataSet(neuro_u32* data, neuro_u32 value, neuro_size_t count) const = 0;
			virtual bool DataSet(neuron_value* data, neuron_value value, neuro_size_t count) const = 0;
			virtual bool Memcpy(void* target, const void* source, neuro_size_t size, math_device_type srctype) const = 0;
			bool Memcpy(void* target, const void* source, neuro_size_t size, const MemoryManager& srcMM) const
			{
				return Memcpy(target, source, size, srcMM.GetType());
			}
		};

		class CPU_MemoryManager : public MemoryManager
		{
		public:
			math_device_type GetType() const override
			{
				return math_device_type::cpu;
			}

			bool GetMemoryInfo(neuro_size_t& free, neuro_size_t& total) const override;

			void* Alloc(void* old, neuro_size_t old_size, neuro_size_t new_size) const override;
			bool Dealloc(void* old) const override;

			bool SetZero(void* data, neuro_size_t size) const override;
			bool DataSet(neuro_u32* data, neuro_u32 value, neuro_size_t count) const override;
			bool DataSet(neuron_value* data, neuron_value value, neuro_size_t count) const override;
			bool Memcpy(void* target, const void* source, neuro_size_t size, math_device_type srctype) const override;
		};

		class CUDA_MemoryManager : public MemoryManager
		{
		public:
			math_device_type GetType() const override
			{
				return math_device_type::cuda;
			}

			bool GetMemoryInfo(neuro_size_t& free, neuro_size_t& total) const override;

			void* Alloc(void* old, neuro_size_t old_size, neuro_size_t new_size) const override;
			bool Dealloc(void* old) const  override;

			bool SetZero(void* data, neuro_size_t size) const override;
			bool DataSet(neuro_u32* data, neuro_u32 value, neuro_size_t count) const override;
			bool DataSet(neuron_value* data, neuron_value value, neuro_size_t count) const override;
			bool Memcpy(void* target, const void* source, neuro_size_t size, math_device_type srctype) const override;
		};
	}
}

#endif
