#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <vector>

enum class GpuType {
    Metal,
    Cuda
};

class GpuScheduler {
    public:
        virtual ~GpuScheduler() = default;
        virtual void* schedule_work() = 0;
};

class MetalScheduler final : public GpuScheduler {
    public:
        MetalScheduler();
        ~MetalScheduler();
        void* schedule_work();
    private:
        MTL::Device* device;
        MTL::CommandQueue* queue;
        std::vector<MTL::ComputePipelineState*> pipelines;
        MTL::ComputePipelineState* build_pipeline(std::vector<char*> name);
};

inline GpuScheduler* make_backend(GpuType t) {
    switch (t) {
        case GpuType::Metal: return new MetalScheduler();
        case GpuType::Cuda:  return new MetalScheduler();
    }
    return nullptr; // should never hit this
}