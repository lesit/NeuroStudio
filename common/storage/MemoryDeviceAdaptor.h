#if !defined(_MEMORY_DEVICE_ADAPTOR_H)
#define _MEMORY_DEVICE_ADAPTOR_H

#include "DeviceAdaptor.h"

namespace np
{
	namespace device
	{
		class MemoryDeviceRefAdaptor : public DeviceAdaptor
		{
		public:
			MemoryDeviceRefAdaptor(neuro_u32 size, void* reference);

			virtual _device_type GetType() const override { return _device_type::memory_reference; }

			// MemoryDeviceAdaptor�� ���� ����ϰ��� �ϸ� �̰ͺ��� �����ؾ���
			virtual std::string GetDeviceName() const override
			{
				return "memory reference";
			}

			std::string GetDevicePath() const override
			{
				return "";
			}

			virtual neuro_u64 GetMaxExtensibleSize() const override { return m_buf_size; }
			virtual neuro_u64 GetUsageSize() const override{ return m_buf_size;	}

			virtual bool SetPosition(neuro_u64 pos) override;
			virtual neuro_u64 GetPosition() const override { return m_position; }
			virtual neuro_u32 Read(void* buffer, neuro_u32 size) const override;
			virtual neuro_u32 Write(const void* buffer, neuro_u32 size) override { return false; }

		protected:
			MemoryDeviceRefAdaptor();

			void* m_buffer;
			neuro_u64 m_buf_size;

			neuro_u64 m_position;
		};

		// memory mapped �� ��������!!
		class MemoryDeviceAdaptor : public MemoryDeviceRefAdaptor
		{
		public:
			MemoryDeviceAdaptor(neuro_u64 init_size);
			virtual ~MemoryDeviceAdaptor();

			virtual _device_type GetType() const {return _device_type::memory;}

			// MemoryDeviceAdaptor�� ���� ����ϰ��� �ϸ� �̰ͺ��� �����ؾ���
			std::string GetDeviceName() const override	
			{
				return "memory";
			}

			virtual neuro_u64 GetMaxExtensibleSize() const override {	// ���� �޸� ���� �뷮�� ���ؾ� ��
				return neuro_last32;
			}

			virtual neuro_u32 Write(const void* buffer, neuro_u32 size) override;
		};

		class MemoryDeviceFactory : public IODeviceFactory
		{
		public:
			MemoryDeviceFactory(){}
			virtual ~MemoryDeviceFactory(){}

			virtual _device_type GetType() const {return _device_type::memory;}

			bool operator == (const IODeviceFactory& right) const
			{
				return right.GetType()== _device_type::memory;
			}

		protected:
			virtual DeviceAdaptor* Create(bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size) override // ���߿� bReadOnly�� ���� ���� ��������.
			{
				return new MemoryDeviceAdaptor(init_size);
			}
		};
	}
}

#endif
