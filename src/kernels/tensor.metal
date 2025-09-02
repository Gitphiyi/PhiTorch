// compiling metal into metal lib
// xcrun -sdk macosx metal -c add.metal -o add.air
// xcrun -sdk macosx metallib add.air -o add.metallib

#include <metal_stdlib>
using namespace metal;

kernel void scalar_mult