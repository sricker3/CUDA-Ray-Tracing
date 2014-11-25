#ifndef CUDA_RAY_TRACING_SPHERE
#define CUDA_RAY_TRACING_SPHERE
#define CUDA_CALL __host__ __device__


#include <vector_types.h>
#include "Ray.h"
#include "Hit.h"
#include "Math.h"

typedef struct sphere
{
    float3 center; //position of the center of the sphere
    float radius;  //radius of the sphere
    float3 color;  //surface color of the sphere
}sphere;

CUDA_CALL bool intersect(sphere& s, hit& h, ray& r)
{
    float3 dir = r.dir;
	float3 org = r.org;
    
	float a = dotProduct(dir, dir);
	float3 oc; oc.x = org.x - s.center.x; oc.y = org.y - s.center.y; oc.z = org.z - s.center.z;
	float b = 2*dotProduct(oc, dir);
	float c = dotProduct(oc, oc) - s.radius*s.radius;
	pairf soln = quadraticSolve(a, b, c);

	float rt = h.t;
    
	if(soln.x1 >= 0 && soln.x1 <= soln.x2 && soln.x1 < rt)
	{
		h.point.x = org.x+dir.x*soln.x1;
		h.point.y = org.y+dir.y*soln.x1;
		h.point.z = org.z+dir.z*soln.x1;
        h.surfaceNormal.x = h.point.x - s.center.x;
        h.surfaceNormal.y = h.point.y - s.center.y;
        h.surfaceNormal.z = h.point.z - s.center.z;
        normalize(h.surfaceNormal);
		h.t = soln.x1;
        h.color = s.color;
		return true;
	}
	
	if(soln.x2 >= 0 && soln.x2 <= soln.x1 && soln.x2 < rt)
	{
		h.point.x = org.x+dir.x*soln.x2;
		h.point.y = org.y+dir.y*soln.x2;
		h.point.z = org.z+dir.z*soln.x2;
        h.surfaceNormal.x = h.point.x - s.center.x;
        h.surfaceNormal.y = h.point.y - s.center.y;
        h.surfaceNormal.z = h.point.z - s.center.z;
        normalize(h.surfaceNormal);
		h.t = soln.x2;
        h.color = s.color;
		return true;
	}

    return false;
}

#endif