
#ifndef CUDA_RAY_TRACING_HIT
#define CUDA_RAY_TRACING_HIT

#include <vector_types.h>

typedef struct hit
{
    float3 point;         //The actual point a ray hit on an object
    float3 surfaceNormal; //The surface normal of the intersected object at the hit point
    float3 color;         //The color of the surface, I think we're using this for now, can remove if not needed
    float t;              //used to tell the distance of the intersection point. Can set to infinity using long num = 0x7f800000; t = *((float*) &num);
}hit;

#endif