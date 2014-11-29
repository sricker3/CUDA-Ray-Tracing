
#include "Math.h"

CUDA_CALL unsigned int randGen(unsigned int seed1, unsigned int seed2)
{
    int random = 0;
    unsigned int w = seed1+125;
    unsigned int z = seed2+30;

    z = 36969 * (z & 65535) + (z >> 16);
    w = 18000 * (w & 65535) + (w >> 16);

    random = (z<<16) + w;

    return random;
}

CUDA_CALL pairf quadraticSolve(float a, float b, float c)
{
	float temp = sqrtf(b*b - 4*a*c);
	pairf ret;
	ret.x1 = (-b+temp)/(2*a);
	ret.x2 = (-b-temp)/(2*a);
	return ret;
}

CUDA_CALL void quaternionRotate(float3 axis, float theta, float3 point)
{
    //theta in radians
    
    float p[4]; p[0] = 0; p[1] = point.x; p[2] = point.y; p[3] = point.z;
    float q[4]; q[0] = cosf(theta/2.0); q[1] = axis.x*sinf(theta/2.0);
    q[2] = axis.y*sinf(theta/2.0); q[3] = axis.z*sinf(theta/2.0);
    float qi[4]; qi[0] = q[0]; qi[1] = -1*q[1]; qi[2] = -1*q[2]; qi[3] = -1*q[3];
    
    float temp[4];
    temp[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3];
    temp[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2];
    temp[2] = q[0]*p[2] + q[2]*p[0] + q[3]*p[1] - q[1]*p[3];
    temp[3] = q[0]*p[3] + q[3]*p[0] + q[1]*p[2] - q[2]*p[1];
    
    float temp2[4];
    temp2[0] = temp[0]*qi[0] - temp[1]*qi[1] - temp[2]*qi[2] - temp[3]*qi[3];
    temp2[1] = temp[0]*qi[1] + temp[1]*qi[0] + temp[2]*qi[3] - temp[3]*qi[2];
    temp2[2] = temp[0]*qi[2] + temp[2]*qi[0] + temp[3]*qi[1] - temp[1]*qi[3];
    temp2[3] = temp[0]*qi[3] + temp[3]*qi[0] + temp[1]*qi[2] - temp[2]*qi[1];
    
    point.x = temp2[1];
    point.y = temp2[2];
    point.z = temp2[3];
    
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

CUDA_CALL void multiply_matrices(float* mat1, float* mat2, float* mat3)
{
    //create temp mats in case mat3 equals mat1 or mat2
    float temp1[16];
    float temp2[16];
    for(int i=0; i<16; i++)
    {
        temp1[i] = mat1[i];
        temp2[i] = mat2[i];
    }
    
    for(int i=0; i<16; i++)
    {
        float sum =0;
        for(int j=0; j<4; j++)
        {
            sum += temp1[(i%4) + 4*j] * temp2[4*(i/4) + j];
        }
        
        mat3[i] = sum;
    }
    
}

CUDA_CALL void translate(float* matrix, float x, float y, float z)
{
    float trans[16]; //could improve run time for this by only multiplying by xyz vector
    trans[0] = 1; trans[4] = 0; trans[8] = 0;   trans[12] = x;
    trans[1] = 0; trans[5] = 1; trans[9] = 0;   trans[13] = y;
    trans[2] = 0; trans[6] = 0; trans[10] = 1;  trans[14] = z;
    trans[3] = 0; trans[7] = 0; trans[11] = 0;  trans[15] = 1;
    
    multiply_matrices(matrix, trans, matrix);
}

CUDA_CALL void multiply_matrix_vector(float* mat, float* vec, float* ret_vec)
{
	if(mat == NULL || vec == NULL || ret_vec == NULL)
		return;
    
	float temp[4]; temp[0] = vec[0]; temp[1] = vec[1]; temp[2] = vec[2]; temp[3] = vec[3];
    
	ret_vec[0] = mat[0]*temp[0] + mat[4]*temp[1] + mat[8]*temp[2] + mat[12]*temp[3];
	ret_vec[1] = mat[1]*temp[0] + mat[5]*temp[1] + mat[9]*temp[2] + mat[13]*temp[3];
	ret_vec[2] = mat[2]*temp[0] + mat[6]*temp[1] + mat[10]*temp[2] + mat[14]*temp[3];
	ret_vec[3] = mat[3]*temp[0] + mat[7]*temp[1] + mat[11]*temp[2] + mat[15]*temp[3];
}