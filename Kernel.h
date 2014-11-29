
#ifndef CUDA_RAY_TRACING_KERNEL
#define CUDA_RAY_TRACING_KERNEL

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <iostream>
#include "Sphere.h"
#include "Triangle.h"
#include "BVHNode.h"
#include "Camera.h"
#include "Math.h"

//using static vars to keep track of gpu pointers
//doesn't seem like best method, may change in future
//these are also only for use by the host as a means
//to pass GPU ptrs to a kernel.

static float3* pixelBufferDevice;
static int dimX;
static int dimY;
static BVHNode* rootDevice;
//these are both technically GPU ptrs, but need a version for both
//as the GPU can't access the one defined in the header for the host.
//if it doesn't stay across kernels, can find another way.
__host__ static curandState* rstate_h;
__device__ static curandState* rstate_d;

//wrapper functions

//maybe don't return node here, just set static one above
__host__ BVHNode* buildBVH(sphere* spheres, int numSphere, triangle* triangles, int numTriangles);
__host__ bool initializePathTracing(float3* pixelBuffer, int x, int y);
__host__ unsigned char* draw(Camera& camera, int maxDepth);

/*
 * A function to stop kernel calls and free any cuda related memory.
 * Call this function when ending the program.
 */
__host__ void finishPathTracing();

#endif