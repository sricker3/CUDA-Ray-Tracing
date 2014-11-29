//
//  camera.h
//
//  Created by Sam Ricker on 6/29/14.
//  Copyright (c) 2014 Sam Ricker. All rights reserved.
//

#ifndef CUDA_RAY_TRACING_CAMERA
#define CUDA_RAY_TRACING_CAMERA
#define CUDA_CALL __host__ __device__

#include <vector_types.h>
#include <iostream>
#include "Math.h"

using namespace std;

class Camera
{
	private:

		float3 lookAt;
		float3 up;
		float3 side;
		float position[4];
		float speed;
		float distToVP;
		float pixelSize;
    
	public:

		CUDA_CALL Camera();
		CUDA_CALL ~Camera();
		CUDA_CALL void lookVertical(float theta);
		CUDA_CALL void lookHorizontal(float theta);
		CUDA_CALL void moveParallel(float increment);
		CUDA_CALL void movePerpendicular(float increment);
		CUDA_CALL void changeSpeed(float increment);
		CUDA_CALL void getViewMatrix(float* viewMat);
        CUDA_CALL void getModViewMatrix(float* viewMat);
		CUDA_CALL void reset();
		CUDA_CALL void setPos(float x, float y, float z);
		CUDA_CALL float* getPos();
        CUDA_CALL float getDistToVP();
        CUDA_CALL float getPixelSize();
        CUDA_CALL float3 getDir();
    
};

#endif /* defined(__Terrain_Demo__camera__) */
