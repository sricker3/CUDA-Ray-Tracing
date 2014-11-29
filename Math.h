
#ifndef CUDA_RAY_TRACING_MATH
#define CUDA_RAY_TRACING_MATH
#define CUDA_CALL __host__ __device__

#include <vector_types.h>

const float RAND_MAX_INV = 1.0/4294967295;

typedef struct pairf
{
	float x1;
	float x2;
}pairf;

CUDA_CALL unsigned int randGen(unsigned int seed1, unsigned int seed2);

CUDA_CALL pairf quadraticSolve(float a, float b, float c);
CUDA_CALL void quaternionRotate(float3 axis, float theta, float3 point);

CUDA_CALL float dotProduct(float3& v1, float3& v2);
CUDA_CALL void normalize(float3& v1);
CUDA_CALL void crossProduct(float3& v1, float3& v2, float3& v3);
CUDA_CALL void crossProductNormalized(float3& v1, float3& v2, float3& v3);
CUDA_CALL void multiply_matrices(float* mat1, float* mat2, float* mat3);
CUDA_CALL void translate(float* matrix, float x, float y, float z);
CUDA_CALL void multiply_matrix_vector(float* mat, float* vec, float* ret_vec);

#endif