
#include "Kernel.h"

/*
 * ====================================
 *  Kernel and Kernel Helper Functions
 * ====================================
 */

__global__ void init_rand(curandState* state, int yl)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y*yl;

    rstate_d = state;
    curand_init(1337+idx, idx, 0, &rstate_d[idx]);
}

/*
 * Place holder
 */
__global__ void BVHBuildKernel()
{

}

__device__ void sample_hemisphere(float3 point)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + dimY*dimX;
    //if any error occurs (ie black screen) check random gen below (2 lines) during debug
    float r1 = curand_uniform(&rstate_d[idx]);
    float r2 = curand_uniform(&rstate_d[idx]); //gens rand# (0.0, 1.0]

    float cos_phi = cos(2.0*3.1415*r1);
    float sin_phi = sin(2.0*3.1415*r1);
    float cos_theta = 1.0-r2;
    float sin_theta = sqrtf(1.0-cos_theta*cos_theta);
    point.x = sin_theta*cos_phi;
    point.y = sin_theta*sin_phi;
    point.z = cos_theta;
}

__device__ float3 BRDF_lamdertian(float3 wo, float3 wi, float3 normal, float3 surface, float& pdf)
{
    float3 t;
    t.x = normal.x+2;
    t.y = normal.y+2;
    t.z = normal.z+2; //the +2 is the lazy way to get a non parallel vector
    normalize(t);
    float3 u;
    float3 v;
    crossProductNormalized(normal, t, u);
    crossProductNormalized(normal, u, v); //getting coord system at surface
    float3 point;
    sample_hemisphere(point);
    wi.x = point.x*v.x + point.y*u.x + point.z*normal.x;
    wi.y = point.x*v.y + point.y*u.y + point.z*normal.y;
    wi.z = point.x*v.z + point.y*u.z + point.z*normal.z;
    normalize(wi);
    pdf = (1/3.1415)*dotProduct(normal, wi);

    float3 ret;
    ret.x = (surface.x/255.0) * 1 * (1/3.1415);
    ret.y = (surface.y/255.0) * 1 * (1/3.1415);
    ret.z = (surface.z/255.0) * 1 * (1/3.1415);
    return ret;
}

__device__ float3 BRDF_specular(float3 wo, float3 wi, float3 normal, float3 surface, float& pdf)
{
    float d = dotProduct(normal, wo);
    wi.x = -wo.x + 2.0*normal.x*d;
    wi.y = -wo.y + 2.0*normal.y*d;
    wi.z = -wo.z + 2.0*normal.z*d;
    pdf = dotProduct(normal, wi);

    float3 ret;
    ret.x = 1;
    ret.y = 1;
    ret.z = 1;
    return ret;
}

/*
 * ===========================
 * Tracing with arrays
 * ===========================
 */

__device__ float3 tracePath(ray r, sphere* spheres, int numSpheres, triangle* triangles, int numTriangles, int maxDepth)
{
    float3 env_light;
    env_light.x = (100/255.0);
    env_light.y = (100/255.0);
    env_light.z = (100/255.0);
    float3 pixelSum;
    pixelSum.x = 1;
    pixelSum.y = 1;
    pixelSum.z = 1;
    hit lastHit;
    ray currRay = r;

    for(int currDepth=1; currDepth<maxDepth; currDepth++)
    {
        //setup for intersections

        lastHit.point.x = 0;
	    lastHit.point.y = 0;
	    lastHit.point.z = 0;
        lastHit.surfaceNormal.x = 0;
        lastHit.surfaceNormal.y = 1;
        lastHit.surfaceNormal.z = 0;
	    long num = 0x7f800000;
	    lastHit.t = *((float*) &num);;  //this sets t to infinity, useful for checking intersections
        lastHit.color.x = 0;
        lastHit.color.y = 0;
        lastHit.color.z = 0;
        
        bool hitCheck = false;
        for(int i=0; i<numSpheres; i++)
        {
            hitCheck = hitCheck || intersect(spheres[i], lastHit, currRay)
        }
        for(int i=0; i<numTriangles; i++)
        {
            hitCheck = hitCheck || intersect(triangles[i], lastHit, currRay)
        }

        bool hitCheck = intersect(root, currRay, lastHit);

        if(hitCheck == NULL)
        {
            float3 bgColor; bgColor.x = 1; bgColor.y = 1; bgColor.z = 1;
            if(currDepth == 1)
            {
                return bgColor;
                //can simply return here as there is nothing to accumulate.
            }
            else
            {
                pixelSum.x *= env_light.x;
                pixelSum.y *= env_light.y;
                pixelSum.z *= env_light.z;
                return pixelSum;
            }
        }

        float3 wi;
        float3 wo;
        wo.x = -r.dir.x;
        wo.y = -r.dir.y;
        wo.z = -r.dir.z;
        float pdf;
        float3 color_brdf;

        //use whatever brdf we want here, could have objects store a function ptr
        //to avoid using if statements (that could be hard though b/c they'd
        //need to be gpu ptrs).

        color_brdf = BRDF_lamdertian(wo, wi, lastHit.surfaceNormal, lastHit.color, pdf);

        float d = dotProduct(lastHit.surfaceNormal, wi);
        currRay.dir.x = wi.x; currRay.dir.y = wi.y; currRay.dir.z = wi.z;
        currRay.org.x = lastHit.point.x+.001*currRay.dir.x; 
        currRay.org.y = lastHit.point.y+.001*currRay.dir.y; 
        currRay.org.z = lastHit.point.z+.001*currRay.dir.z;

        //colorf ret;
        //colorf childColor = tracePath(childRay, world, maxDepth, currDepth+1);
        pixelSum.x *= color_brdf.x * (d/pdf);
        pixelSum.y *= color_brdf.y * (d/pdf);
        pixelSum.z *= color_brdf.z * (d/pdf);
    }

    //if we got here, we can assume that we need to mult by env light

    pixelSum.x *= env_light.x;
    pixelSum.y *= env_light.y;
    pixelSum.z *= env_light.z;
    return pixelSum;
}

__global__ void pathTrace(sphere* spheres, int numSpheres, triangle* triangles, int numTriangles,, Camera camera, float3* pixelBuffer, int dimX, int dimY, int maxDepth)
{
//pixel coordinates this thread will compute
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

    //calc multisampling points
    int N = 36;
    float points[72]; //N & points set manually due to cuda complications (just easier this way)
    int sqrtN = (int) sqrtf(N);
    float invSqrtN = 1.0/sqrtN;
    float s = camera.getPixelSize();
    float dvp = camera.getDistToVP();

    //setup jittered grid with the rook condition
    for(int row=0; row<sqrtN; row++)
    {
        for(int col=0; col<sqrtN; col++)
        {
            float rand1 = randGen(row, col)*RAND_MAX_INV;
            float rand2 = randGen(col, row)*RAND_MAX_INV;
            points[2*(sqrtN*row + col)] = (col + (row+rand1)*invSqrtN)*invSqrtN;
            points[2*(sqrtN*row + col)+1] = (row + (col+rand2)*invSqrtN)*invSqrtN;
        }
    }

    //randomly shuffle
    for(int row=0; row<sqrtN; row++)
    {
        unsigned int rand = randGen(row*977, row*7) % (sqrtN - 1);
        for(int col=0; col<sqrtN; col++)
        {
            float temp = points[2*(sqrtN*row + col)];
            points[2*(sqrtN*row + col)] = points[2*(sqrtN*rand + col)];
            points[2*(sqrtN*rand + col)] = temp;
        }
    }

    for(int col=0; col<sqrtN; col++)
    {
        unsigned int rand = randGen(col*7, col*379) % (sqrtN - 1);
        for(int row=0; row<sqrtN; row++)
        {
            float temp = points[2*(sqrtN*row + col)+1];
            points[2*(sqrtN*row + col)+1] = points[2*(sqrtN*row + rand)+1];
            points[2*(sqrtN*row + rand)+1] = temp;
        }
    }

    //end calc multisampling points

    //call helper function for each primary ray sample

    float3 pixelSum;
    pixelSum.x = 0;
    pixelSum.y = 0;
    pixelSum.z = 0;
    for(int i=0; i<N; i++)
    {
        float dir[4];
        dir[0] = s*(x - (dimX/2.0) + points[2*i]); 
        dir[1] = s*(y - (dimY/2.0) + points[(2*i)+1]); 
        dir[2] = -dvp; 
        dir[3] = 1;
        float viewMat[16];
        camera.getModViewMatrix(viewMat);
        multiply_matrix_vector(viewMat, dir, dir);
        ray currRay;
        currRay.org.x = camera.getPos()[0]; 
        currRay.org.y = camera.getPos()[1]; 
        currRay.org.z = camera.getPos()[2];
        currRay.dir.x = dir[0]; 
        currRay.dir.y = dir[1]; 
        currRay.dir.z = dir[2];
        float3 val = tracePath(currRay, spheres, numSpheres, triangles, numTriangles, maxDepth); 
        pixelSum.x += val.x;
	    pixelSum.y += val.y;
        pixelSum.z += val.z;

        //update pixelBuffer so we can grab a value if kernel isn't fast enough
        //not sure how costly this will be compared to other methods but it's
        //easy to remove if needed

        pixelBuffer[(x + (dimY-1-y)*dimX)].x = pixelSum.x/(i+1);
	    pixelBuffer[(x + (dimY-1-y)*dimX)].y = pixelSum.y/(i+1);
        pixelBuffer[(x + (dimY-1-y)*dimX)].z = pixelSum.z/(i+1);
    }

    pixelSum.x /= N;
    pixelSum.y /= N;
    pixelSum.z /= N;

    pixelBuffer[(x + (dimY-1-y)*dimX)].x = pixelSum.x;
	pixelBuffer[(x + (dimY-1-y)*dimX)].y = pixelSum.y;
    pixelBuffer[(x + (dimY-1-y)*dimX)].z = pixelSum.z;
}

/*
 * ===========================
 * Tracing with the BVH tree
 * ===========================
 */

__device__ float3 tracePath(ray r, BVHNode* root, int maxDepth)
{
    float3 env_light;
    env_light.x = (100/255.0);
    env_light.y = (100/255.0);
    env_light.z = (100/255.0);
    float3 pixelSum;
    pixelSum.x = 1;
    pixelSum.y = 1;
    pixelSum.z = 1;
    hit lastHit;
    ray currRay = r;

    for(int currDepth=1; currDepth<maxDepth; currDepth++)
    {
        //setup for intersections

        lastHit.point.x = 0;
	    lastHit.point.y = 0;
	    lastHit.point.z = 0;
        lastHit.surfaceNormal.x = 0;
        lastHit.surfaceNormal.y = 1;
        lastHit.surfaceNormal.z = 0;
	    long num = 0x7f800000;
	    lastHit.t = *((float*) &num);;  //this sets t to infinity, useful for checking intersections
        lastHit.color.x = 0;
        lastHit.color.y = 0;
        lastHit.color.z = 0;

        bool hitCheck = intersect(root, currRay, lastHit);

        if(hitCheck == NULL)
        {
            float3 bgColor; bgColor.x = 1; bgColor.y = 1; bgColor.z = 1;
            if(currDepth == 1)
            {
                return bgColor;
                //can simply return here as there is nothing to accumulate.
            }
            else
            {
                pixelSum.x *= env_light.x;
                pixelSum.y *= env_light.y;
                pixelSum.z *= env_light.z;
                return pixelSum;
            }
        }

        float3 wi;
        float3 wo;
        wo.x = -r.dir.x;
        wo.y = -r.dir.y;
        wo.z = -r.dir.z;
        float pdf;
        float3 color_brdf;

        //use whatever brdf we want here, could have objects store a function ptr
        //to avoid using if statements (that could be hard though b/c they'd
        //need to be gpu ptrs).

        color_brdf = BRDF_lamdertian(wo, wi, lastHit.surfaceNormal, lastHit.color, pdf);

        float d = dotProduct(lastHit.surfaceNormal, wi);
        currRay.dir.x = wi.x; currRay.dir.y = wi.y; currRay.dir.z = wi.z;
        currRay.org.x = lastHit.point.x+.001*currRay.dir.x; 
        currRay.org.y = lastHit.point.y+.001*currRay.dir.y; 
        currRay.org.z = lastHit.point.z+.001*currRay.dir.z;

        //colorf ret;
        //colorf childColor = tracePath(childRay, world, maxDepth, currDepth+1);
        pixelSum.x *= color_brdf.x * (d/pdf);
        pixelSum.y *= color_brdf.y * (d/pdf);
        pixelSum.z *= color_brdf.z * (d/pdf);
    }

    //if we got here, we can assume that we need to mult by env light

    pixelSum.x *= env_light.x;
    pixelSum.y *= env_light.y;
    pixelSum.z *= env_light.z;
    return pixelSum;
}

__global__ void pathTrace(BVHNode* root, Camera camera, float3* pixelBuffer, int dimX, int dimY, int maxDepth)
{
//pixel coordinates this thread will compute
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

    //calc multisampling points
    int N = 36;
    float points[72]; //N & points set manually due to cuda complications (just easier this way)
    int sqrtN = (int) sqrtf(N);
    float invSqrtN = 1.0/sqrtN;
    float s = camera.getPixelSize();
    float dvp = camera.getDistToVP();

    //setup jittered grid with the rook condition
    for(int row=0; row<sqrtN; row++)
    {
        for(int col=0; col<sqrtN; col++)
        {
            float rand1 = randGen(row, col)*RAND_MAX_INV;
            float rand2 = randGen(col, row)*RAND_MAX_INV;
            points[2*(sqrtN*row + col)] = (col + (row+rand1)*invSqrtN)*invSqrtN;
            points[2*(sqrtN*row + col)+1] = (row + (col+rand2)*invSqrtN)*invSqrtN;
        }
    }

    //randomly shuffle
    for(int row=0; row<sqrtN; row++)
    {
        unsigned int rand = randGen(row*977, row*7) % (sqrtN - 1);
        for(int col=0; col<sqrtN; col++)
        {
            float temp = points[2*(sqrtN*row + col)];
            points[2*(sqrtN*row + col)] = points[2*(sqrtN*rand + col)];
            points[2*(sqrtN*rand + col)] = temp;
        }
    }

    for(int col=0; col<sqrtN; col++)
    {
        unsigned int rand = randGen(col*7, col*379) % (sqrtN - 1);
        for(int row=0; row<sqrtN; row++)
        {
            float temp = points[2*(sqrtN*row + col)+1];
            points[2*(sqrtN*row + col)+1] = points[2*(sqrtN*row + rand)+1];
            points[2*(sqrtN*row + rand)+1] = temp;
        }
    }

    //end calc multisampling points

    //call helper function for each primary ray sample

    float3 pixelSum;
    pixelSum.x = 0;
    pixelSum.y = 0;
    pixelSum.z = 0;
    for(int i=0; i<N; i++)
    {
        float dir[4];
        dir[0] = s*(x - (dimX/2.0) + points[2*i]); 
        dir[1] = s*(y - (dimY/2.0) + points[(2*i)+1]); 
        dir[2] = -dvp; 
        dir[3] = 1;
        float viewMat[16];
        camera.getModViewMatrix(viewMat);
        multiply_matrix_vector(viewMat, dir, dir);
        ray currRay;
        currRay.org.x = camera.getPos()[0]; 
        currRay.org.y = camera.getPos()[1]; 
        currRay.org.z = camera.getPos()[2];
        currRay.dir.x = dir[0]; 
        currRay.dir.y = dir[1]; 
        currRay.dir.z = dir[2];
        float3 val = tracePath(currRay, root, maxDepth); 
        pixelSum.x += val.x;
	    pixelSum.y += val.y;
        pixelSum.z += val.z;

        //update pixelBuffer so we can grab a value if kernel isn't fast enough
        //not sure how costly this will be compared to other methods but it's
        //easy to remove if needed

        pixelBuffer[(x + (dimY-1-y)*dimX)].x = pixelSum.x/(i+1);
	    pixelBuffer[(x + (dimY-1-y)*dimX)].y = pixelSum.y/(i+1);
        pixelBuffer[(x + (dimY-1-y)*dimX)].z = pixelSum.z/(i+1);
    }

    pixelSum.x /= N;
    pixelSum.y /= N;
    pixelSum.z /= N;

    pixelBuffer[(x + (dimY-1-y)*dimX)].x = pixelSum.x;
	pixelBuffer[(x + (dimY-1-y)*dimX)].y = pixelSum.y;
    pixelBuffer[(x + (dimY-1-y)*dimX)].z = pixelSum.z;
}

/*
 * ===============
 *  HOST WRAPPERS
 * ===============
 */

/*
 * Place holder
 */
__host__ BVHNode* buildBVH(sphere* spheres, int numSphere, triangle* triangles, int numTriangles)
{
    //call kernel function above and build tree on GPU?
    //Return GPU pointer so it can be used in drawing (could possibly save it as a static var in header
    //use ***cudaThreadSynchronize()*** to block the host until the kernel has finished (want the tree to be 
    //completly built before using it).
}

/*
 * Call this function once to setup data for the GPU. This function should always
 * be called before any calls to draw.
 * -The function expects a ptr that it can allocate and use to setup a gpu ptr
 * -The pixelBuffer is for storing pixel/color values, assuming RGB.
 * -@return: returns a bool indicating success (false for failure)
 * -Scratched (removed ideas):
 *      -The rayBuffer is for storing the next rays on the path to be processed. This is a device var.
 */
__host__ bool initializePathTracing(float3* pixelBuffer, int x, int y)
{
    dimX = x;
    dimY = y;

    if(cudaHostAlloc((void**)&pixelBuffer, sizeof(float3)*x*y, cudaHostAllocMapped))
    {
        return false;
    }

    if(cudaHostGetDevicePointer((void**)&pixelBufferDevice, (void*)pixelBuffer, 0))
    {
        return false;
    }

    if(cudaMalloc((void**)&rstate_h, x*y))
    {
        return false;
    }

    dim3 blocks(x/32, y/16); //can bump 16 to 32 if we want
	dim3 threads(32, 16);
    init_rand<<<blocks, threads>>>(rstate_h, dimY);

    cudaError err = cudaGetLastError();
    if(err)
    {
        std::cout<<cudaGetErrorString(err)<<std::endl;
        return false;
    }

    return true;
}

__host__ void giveObjects(sphere* s, int ns, triangle* t, int nt)
{
    numSpheres = ns;
    numTriangles = nt;
    
    cudaMalloc((void**)&spheres, sizeof(sphere)*ns);
    cudaMalloc((void**)&triangles, sizeof(triangle)*nt);
    cudaMemcpy(spheres, s, sizeof(sphere)*ns);
    cudaMemcpy((triangles, t, sizeof(triangle)*nt);
}

unsigned char* draw(Camera camera, int maxDepth)
{
    //maybe try this asynchronously? this starts the drawing, another function grabs memory?
    
    //use cudaStreamQuery() to force start a kernel (needed with mappedMem)
    //use non-default stream if mapped mem fails

    dim3 blocks(x/32, y/16); //can bump 16 to 32 if we want
	dim3 threads(32, 16);
    pathTrace<<<blocks, threads>>>(spheres, numSpheres, triangles, numTriangles, camera, pixelBufferDevice, dimX, dimY, maxDepth);
    cudaStreamQuery(0);

}

__host__ void finishPathTracing()
{
    cudaDeviceSynchronize();
    cudaFreeHost(pixelBufferDevice);
    cudaFree(rstate_h);
    cudaFree(spheres);
    cudaFree(triangles);
    cudaDeviceReset();
}
