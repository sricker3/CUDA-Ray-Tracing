
#ifndef CUDA_RAY_TRACER_RAY
#define CUDA_RAY_TRACER_RAY

#include <vector_types.h>

typedef struct ray
{
    float3 org; //ray's origin point
    float3 dir; //ray's direction vector
    //unsigned char depth; //used to keep track of rays depth from 0 to maxDepth
                         //perhaps just use an int as it might use 4 bytes anyways due to packing
}ray;

#endif