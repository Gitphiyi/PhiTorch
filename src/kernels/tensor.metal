// compiling metal into metal lib
// xcrun -sdk macosx metal -c add.metal -o add.air
// xcrun -sdk macosx metallib add.air -o add.metallib

#include <metal_stdlib>
using namespace metal;

// A: [M x N], row-major
// B: [N x M], row-major (output)
kernel void transpose_naive(device const float* A   [[buffer(0)]],
                            device float*       B   [[buffer(1)]],
                            constant uint2&     MN  [[buffer(2)]], 
                            uint2 gid [[thread_position_in_grid]]) {

    const uint M = MN.x; // rows of A
    const uint N = MN.y; // cols of A

    if (gid.x >= N || gid.y >= M) return;

    const uint i = gid.y; // row in A
    const uint j = gid.x; // col in A

    // A[i,j] transposes to B[j,i]
    B[j * M + i] = A[i * N + j];
}