#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>

__global__ void accelComputeKernal(vector3** dev_accels, double * dev_mass, vector3* dev_hPos, vector3* dev_values){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = threadIdx.z;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		dev_accels[i]=&dev_values[i*NUMENTITIES];

		if (i==j) {
			FILL_VECTOR(dev_accels[i][j],0,0,0);
		}else{
			vector3 distance;
			distance[k]= dev_hPos[i][k] - dev_hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*dev_mass[j]/magnitude_sq;
			FILL_VECTOR(dev_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
		}
	}
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3** dev_accels;
	double * dev_mass;
	vector3* dev_hPos;
	vector3* dev_values;
	cudaMalloc(&dev_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMalloc(&dev_accels, sizeof(vector3*) * NUMENTITIES);
	cudaMalloc(&dev_mass, sizeof(double) * NUMENTITIES );
	cudaMemcpy(dev_mass, mass, sizeof(double) * NUMENTITIES,cudaMemcpyHostToDevice);
	cudaMalloc(&dev_hPos, sizeof(vector3) * NUMENTITIES );
	cudaMemcpy(dev_hPos, hPos,sizeof(vector3) * NUMENTITIES,cudaMemcpyHostToDevice);
	
	dim3 blockSize(18,18,3);
	dim3 numBlocks((NUMENTITIES+323)/324);
	
	accelComputeKernal<<<numBlocks, blockSize>>>(dev_accels, dev_mass, dev_hPos, dev_values);
	cudaError_t cudaError = cudaGetLastError();

	vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
	cudaMemcpy(accels, dev_accels, sizeof(vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(cudaError));
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	cudaFree(dev_hPos);
	cudaFree(dev_mass);
	cudaFree(dev_accels);
	cudaFree(dev_values);
}
