#pragma once

#include "../np_types.h"

namespace np
{
	namespace device
	{
		enum class _device_type
		{
			file, 
			memory,
			memory_reference,
			network,
		};

		class DeviceAdaptor
		{
		public:
			DeviceAdaptor(){};
			virtual ~DeviceAdaptor(){};

			virtual _device_type GetType() const = 0;

			virtual std::string GetDeviceName() const = 0;	// linux호환을 위해 utf8로 한다.
			virtual std::string GetDevicePath() const = 0;

			virtual bool IsFixedDevice() const { return false; }	// HDD/SSD 등 고정 장치면 true를 리턴한다.
			virtual neuro_u64 GetMaxExtensibleSize() const = 0;

			virtual bool SetUsageSize(neuro_u64 new_size){return true;};
			virtual neuro_u64 GetUsageSize() const = 0;

			virtual bool SetPosition(neuro_u64 pos) = 0;
			virtual neuro_u64 GetPosition() const = 0;
			virtual neuro_u32 Read(void* buffer, neuro_u32 nSize) const = 0;
			virtual neuro_u32 Write(const void* buffer, neuro_u32 nSize) = 0;

			virtual bool Flush() { return true; }

			neuro_u32 ref_count;	// 나중에 이걸 통해서 정리 할 수 있도록 하자
		};

		class IODeviceFactory
		{
		public:
			virtual ~IODeviceFactory(){};

			virtual _device_type GetType() const = 0;

			virtual void Reset(){};

			DeviceAdaptor* CreateWriteAdaptor(bool bCreateAlways, bool bShareWrite, neuro_u64 init_size = 0)
			{
				return Create(false, bCreateAlways, bShareWrite, init_size);
			}

			DeviceAdaptor* CreateReadOnlyAdaptor()
			{
				return Create(true, false, false, 0);
			}

			virtual bool operator == (const IODeviceFactory& right) const = 0;

		protected:
			virtual DeviceAdaptor* Create(bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size) = 0;
		};
	}
}
