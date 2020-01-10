#include <iostream>
#include <cuda_runtime.h>

void print_cuda_devices()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "--- Device No.: " << i << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "PCI Bus::Device::Domain: " << prop.pciBusID << "::" << prop.pciDeviceID << "::" << prop.pciDeviceID << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "GPU Clock Rate (GHz): " << prop.clockRate/1000.0/1000.0 << std::endl;
        std::cout << "Total Global Memory (MiB): " << prop.totalGlobalMem/1024/1024 << std::endl;
        std::cout << "L2 Cache Size (KiB): " << prop.l2CacheSize/1024 << std::endl;
    }
}

int main()
{
        print_cuda_devices();
        return 1;
}
