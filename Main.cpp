#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <iostream>
#include "Kernel.h"
#include "Sphere.h"
#include "Triangle.h"

int main(int argc, char** argv)
{
    //Print GPU name for fun
    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);
    std::cout<<"GPU name: "<<device.name<<std::endl;
    //Think this is used for mapped memory, will remove if not.
    std::cout<<cudaSetDeviceFlags(cudaDeviceMapHost)<<std::endl;

    /*
     * Use Assimp for model loading here when we want to add that in
     */

    sphere* spheres = NULL;
    int numSpheres = 0;
    triangle* triangles = NULL;
    int numTriangles = 0;
    buildBVH(spheres, numSpheres, triangles, numTriangles);

    float3* pixelBuffer;
    int x = 512;
    int y = 512; //image size, can change, but should try to keep as a square w/ power of 2 side length
    if(!initializePathTracing(pixelBuffer, x, y))
    {
        std::cout<<"unable to initialize the GPU, quitting the program\n";
        return 1;
    }

    //start main loop here
    //are we using GLUT? can't remember, but this should be where
    //key input and repetitive drawing stuff goes

    //before ending, call closing function

    finishPathTracing();
    
}