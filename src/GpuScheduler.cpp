#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "GpuScheduler.hpp"

#include <iostream>
#include <fstream>

using namespace std;

MetalScheduler::MetalScheduler() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::cerr << "No Metal device.\n"; }
    MTL::CommandQueue* queue = device->newCommandQueue();

    //     MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &err);
    // if (!pso) {
    //     std::cerr << "PSO error: " << (err ? err->localizedDescription()->utf8String() : "unknown") << "\n";
    // }
}
MetalScheduler::~MetalScheduler() {
    if (queue)  { queue->release();  queue  = nullptr; }
    if (device) { device->release(); device = nullptr; }
}

void* MetalScheduler::schedule_work() {
    // example no-op command buffer
    auto* cb = queue->commandBuffer();
    cb->commit();
    cb->waitUntilCompleted();
    cb->release();
    return nullptr;
}


// MTL::ComputePipelineState* MetalScheduler::build_pipeline(const string* name) {
//     for (auto name : kernelNames) {
//     NS::Error* err = nullptr;

//     auto fnName = NS::String::string(name, NS::UTF8StringEncoding);
//     MTL::Function* fn = library_->newFunction(fnName);
//     assert(fn && "Kernel function not found in library");

//     // Synchronous compile; switch to the async overload if you want to prewarm in parallel.
//     MTL::ComputePipelineState* pso = device_->newComputePipelineState(fn, &err);
//     fn->release();
//     assert(pso && "Failed to create compute pipeline state");

//     computePSO_.emplace(name, pso);
//   }
// }