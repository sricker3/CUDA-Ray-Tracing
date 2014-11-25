
#include "Math.h"

CUDA_CALL pairf quadraticSolve(float a, float b, float c)
{
	float temp = sqrtf(b*b - 4*a*c);
	pairf ret;
	ret.x1 = (-b+temp)/(2*a);
	ret.x2 = (-b-temp)/(2*a);
	return ret;
}

CUDA_CALL float dotProduct(float3& v1, float3& v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

CUDA_CALL void normalize(float3& v1)
{
    if(v1.x == 0 && v1.y == 0 && v1.z == 0)
        return;
    
	float length = sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
    
    v1.x /= length;
    v1.y /= length;
    v1.z /= length;
    
    if(abs(v1.x) < .000001)
        v1.x = 0;
    if(abs(v1.y) < .000001)
        v1.y = 0;
    if(abs(v1.z) < .000001)
        v1.z = 0;
    
}

CUDA_CALL void crossProduct(float3& v1, float3& v2, float3& v3)
{
    v3.x = v1.y*v2.z - v2.y*v1.z;
    v3.y = -1 * (v1.x*v2.z - v2.x*v1.z);
    v3.z = v1.x*v2.y - v2.x*v1.y;
    
}

CUDA_CALL void crossProductNormalized(float3& v1, float3& v2, float3& v3)
{
	v3.x = v1.y*v2.z - v2.y*v1.z;
	v3.y = -1 * (v1.x*v2.z - v2.x*v1.z);
	v3.z = v1.x*v2.y - v2.x*v1.y;
    
    float length = sqrt(v3.x*v3.x + v3.y*v3.y + v3.z*v3.z);
    
    v3.x /= length;
    v3.y /= length;
    v3.z /= length;
}