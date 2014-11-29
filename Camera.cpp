//
//  camera.cpp
//
//  Created by Sam Ricker on 6/29/14.
//  Copyright (c) 2014 Sam Ricker. All rights reserved.
//

//use left handed camera

#include "Camera.h"

CUDA_CALL Camera::Camera()
{
    
	lookAt.x = 0;  //0
    lookAt.y = 0;  //0
    lookAt.z = -1; //-1
    
    position[0] = 0; //0
    position[1] = 1; //0
    position[2] = -1; //0
    position[3] = 1;
    
    //camera's +y azis
    up.x = 0;
    up.y = 1;
    up.z = 0;
    
    //camera's +x axis
    crossProductNormalized(up, lookAt, side);
    speed = .01;
	distToVP = 1;
	pixelSize = 1.0/512.0;
}

CUDA_CALL Camera::~Camera()
{

}

CUDA_CALL void Camera::lookVertical(float theta)
{
    quaternionRotate(side, theta, lookAt);
    quaternionRotate(side, theta, up);
}

CUDA_CALL void Camera::lookHorizontal(float theta)
{
    float3 y; y.x = 0; y.y = 1; y.z = 0;
    quaternionRotate(y, theta, lookAt);
    quaternionRotate(y, theta, up);
    quaternionRotate(y, theta, side);
}

CUDA_CALL void Camera::moveParallel(float increment)
{
	position[0] += speed * increment * lookAt.x;
	position[1] += speed * increment * lookAt.y;
	position[2] += speed * increment * lookAt.z;
}

CUDA_CALL void Camera::movePerpendicular(float increment)
{
	position[0] += speed * increment * side.x;
	position[1] += speed * increment * side.y;
	position[2] += speed * increment * side.z;
}

CUDA_CALL void Camera::changeSpeed(float increment)
{
	speed += increment;
    if(speed < 0)
        speed = 0;
    
}

CUDA_CALL void Camera::getViewMatrix(float* viewMat)
{
    if(viewMat == NULL)
        return;
    viewMat[0] = side.x;    viewMat[1] = side.y;    viewMat[2] = side.z;     viewMat[3] = 0;
	viewMat[4] = up.x;      viewMat[5] = up.y;      viewMat[6] = up.z;       viewMat[7] = 0;
	viewMat[8] = lookAt.x;  viewMat[9] = lookAt.y;  viewMat[10] = lookAt.z;  viewMat[11] = 0;
    viewMat[12] = 0;        viewMat[13] = 0;        viewMat[14] = 0;         viewMat[15] = 1;
    
    translate(viewMat, -position[0], -position[1], -position[2]);
}

CUDA_CALL void Camera::getModViewMatrix(float* viewMat)
{
    if(viewMat == NULL)
        return;
    viewMat[0] = side.x;    viewMat[1] = side.y;    viewMat[2] = side.z;     viewMat[3] = 0;
	viewMat[4] = up.x;      viewMat[5] = up.y;      viewMat[6] = up.z;       viewMat[7] = 0;
	viewMat[8] = lookAt.x;  viewMat[9] = lookAt.y;  viewMat[10] = lookAt.z;  viewMat[11] = 0;
    viewMat[12] = 0;        viewMat[13] = 0;        viewMat[14] = 0;         viewMat[15] = 1;
}

CUDA_CALL void Camera::reset()
{
	lookAt.x = 0;
    lookAt.y = 0;
    lookAt.z = -1;
    
    position[0] = 0;
    position[1] = .25;
    position[2] = 1;
    position[3] = 1;
    
    //camera's +y azis
    up.x = 0;
    up.y = 1;
    up.z = 0;
    
    //camera's +x axis
    crossProductNormalized(up, lookAt, side);
    
    speed = .01;
}

CUDA_CALL void Camera::setPos(float x, float y, float z)
{
    position[0] = x;
    position[1] = y;
    position[2] = z;
}

CUDA_CALL float* Camera::getPos()
{
	return position;
}

CUDA_CALL float Camera::getDistToVP()
{
    return distToVP;
}

CUDA_CALL float Camera::getPixelSize()
{
    return pixelSize;
}

CUDA_CALL float3 Camera::getDir()
{
    return lookAt;
}