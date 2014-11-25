
#ifndef CUDA_RAY_TRACING_MATH
#define CUDA_RAY_TRACING_MATH
#define CUDA_CALL __host__ __device__

#include <vector_types.h>

typedef struct pairf
{
	float x1;
	float x2;
}pairf;

CUDA_CALL pairf quadraticSolve(float a, float b, float c);
CUDA_CALL float dotProduct(float3& v1, float3& v2);
CUDA_CALL void normalize(float3& v1);
CUDA_CALL void crossProduct(float3& v1, float3& v2, float3& v3);
CUDA_CALL void crossProductNormalized(float3& v1, float3& v2, float3& v3);

#endif