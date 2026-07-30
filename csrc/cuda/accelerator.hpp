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
    printf("cudaGetDeviceCount: err=%d (%s), count=%d\n", (int)err, cudaGetErrorString(err), count);
    fflush(stdout);
    return (err == cudaSuccess) && (count > 0);
#else   
    printf("DLF_CUDA not defined at compile time\n");
    fflush(stdout);
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
                if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
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