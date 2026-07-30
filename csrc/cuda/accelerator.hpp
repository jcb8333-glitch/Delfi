#pragma once

#include <string>
#include <vector>
#ifdef DLF_CUDA
    #include <cuda_runtime.h>
#endif

enum class DeviceType {
    CPU,
    CUDA
};

struct DeviceInfo{
    int index;
    std::string name;
    size_t totalMemory;
};

class Accelerator {
    public:
        static bool available(){
#ifdef DLF_CUDA
            int count = 0;
            cudaError_t err = cudaGetDeviceCount(&count);
            return (err == cudaSuccess) && (count > 0);
#else
            return false;
#endif
        }

        static std::vector<DeviceInfo> listDevices(){
            std::vector<DeviceInfo> devices;
#ifdef DLF_CUDA
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess) return devices;
            for (int i = 0; i < count; ++i){
                cudaDeviceProp prop;
                if (cudaDeviceProperies(&prop, i) == cudaSuccess) {
                    devices.push_back({i, std::string(prop.name), prop.totalGlobalMem});
                }
            }
#endif
            return devices;
        }

        static DeviceType defaultDevice(){
            return available() ? DeviceType::CUDA : DeviceType::CPU;
        }
};