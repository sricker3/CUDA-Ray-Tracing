
#ifndef CUDA_RAY_TRACER_RAY
#define CUDA_RAY_TRACER_RAY

#include <vector_types.h>

typedef struct ray
{
    float3 org; //ray's origin point
    float3 dir; //ray's direction vector
}ray;

#endif