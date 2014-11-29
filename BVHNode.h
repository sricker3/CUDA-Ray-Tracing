
/*
 * NOTE: This is header file is a place holder for whatever we end up using
 * for our BVH structure. Replace or fill in as needed.
 */

#ifndef CUDA_RAY_TRACER_BVH
#define CUDA_RAY_TRACER_BVH
#define CUDA_CALL __host__ __device__

typedef struct BVHNode
{
    int nodeID; //some random variable;
}BVHNode;

CUDA_CALL bool intersect(BVHNode* root, ray& r, hit& h)
{
    //something goes here
    return false;
}

#endif