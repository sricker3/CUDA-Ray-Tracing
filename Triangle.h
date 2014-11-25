
#ifndef CUDA_RAY_TRACING_TRIANGLE
#define CUDA_RAY_TRACING_TRIANGLE
#define CUDA_CALL __host__ __device__


#include <vector_types.h>
#include "Ray.h"
#include "Hit.h"
#include "Math.h"

typedef struct triangle
{
    float3 p0;
    float3 p1;
    float3 p2;
    float3 color;
    //normal?
    //per vertex normals?
    //area?
}triangle;

CUDA_CALL bool intersect(triangle& tri, ray& r, hit& h)
{
    //get the triangle's normal (assumes points defined in counter clockwise fashion) 
    float3 v1; v1.x = tri.p1.x - tri.p0.x; v1.y = tri.p1.y - tri.p0.y; v1.z = tri.p1.z - tri.p0.z;
	float3 v2; v2.x = tri.p2.x - tri.p0.x; v2.y = tri.p2.y - tri.p0.y; v2.z = tri.p2.z - tri.p0.z;
    float3 normal;
	crossProductNormalized(v1, v2, normal);

    //get triangle's area

	float3 c1;
	crossProduct(v1, v2, c1);
	float area = .5f*sqrtf(c1.x*c1.x + c1.y*c1.y + c1.z*c1.z);

	float3 dir = r.dir;
	float denom = dotProduct(dir, normal);
	if(-.0000001 < denom && denom < .0000001)
		return false;

	//get intersection point
	float3 org = r.org;
	float3 ao; ao.x = tri.p0.x - org.x; ao.y = tri.p0.y - org.y; ao.z = tri.p0.z - org.z;
	float numor = dotProduct(ao, normal);
	float t = numor/denom;
	if(t < 0)
		return false;

	float3 point; point.x = org.x + dir.x*t; point.y = org.y + dir.y*t; point.z = org.z + dir.z*t;

	//test against barycentric coordinates
    //Need signed area!! Check by dot product of sub triangle normal with triangle normal, if negative
    //p is not in triangle

	float3 p0p; p0p.x = tri.p0.x - point.x; p0p.y = tri.p0.y - point.y; p0p.z = tri.p0.z - point.z;
	float3 p1p; p1p.x = tri.p1.x - point.x; p1p.y = tri.p1.y - point.y; p1p.z = tri.p1.z - point.z;
	float3 p2p; p2p.x = tri.p2.x - point.x; p2p.y = tri.p2.y - point.y; p2p.z = tri.p2.z - point.z;

	float3 c0; float3 c1; float3 c2;
	crossProduct(p1p, p2p, c0); crossProduct(p2p, p0p, c1); crossProduct(p0p, p1p, c2);
    float dot0 = dotProduct(c0, normal);
    float dot1 = dotProduct(c1, normal);
    float dot2 = dotProduct(c2, normal);
    bool negDot = false;
    if(dot0 < 0 || dot1 < 0 || dot2 < 0)
        negDot = true;

	float a0 = .5f*sqrtf(c0.x*c0.x + c0.y*c0.y + c0.z*c0.z);
	float a1 = .5f*sqrtf(c1.x*c1.x + c1.y*c1.y + c1.z*c1.z);
	float a2 = .5f*sqrtf(c2.x*c2.x + c2.y*c2.y + c2.z*c2.z);
	float b0 = a0 / area;
	float b1 = a1 / area;
	float b2 = a2 / area;

	if(0 <= b0 && b0 <= 1 && 0 <= b1 && b1 <= 1 && 0 <= b2 && b2 <= 1 && !negDot &&  t < h.t)
	{
		h.point.x = point.x; h.point.y = point.y; h.point.z = point.z;
        if(denom > 0)
        {
            h.surfaceNormal.x = -normal.x;
            h.surfaceNormal.y = -normal.y;
            h.surfaceNormal.z = -normal.z;
        }
        else
        {
            h.surfaceNormal.x = normal.x;
            h.surfaceNormal.y = normal.y;
            h.surfaceNormal.z = normal.z;
        }
		h.t = t;
        h.color = tri.color;
	}
}

#endif