#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}
extern vector3 *hVel, *dev_hVel;
extern vector3 *hPos, *dev_hPos;
extern double *mass;

extern vector3* dev_accels;
#endif