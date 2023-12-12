#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>

__global__ void accelComputeKernal(vector3* dev_accels, double * dev_mass, vector3* dev_hPos){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i * NUMENTITIES + j;

	if (i < NUMENTITIES && j < NUMENTITIES) {
		if (i==j) {
			FILL_VECTOR(dev_accels[index],0,0,0);
		}else{
			vector3 distance;
			for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
			FILL_VECTOR(accels[index],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);	
		}
	}
}

// __global__ void contructAccels(vector3** dev_accels, vector3* dev_values){
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (i < NUMENTITIES){
// 		dev_accels[i] = &dev_values[i*NUMENTITIES];
// 	}
// }

__global__ void sumRows(vector3* dev_accels, vector3* dev_hPos, vector3* dev_hVel){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = threadIdx.z;

	if(i < NUMENTITIES && j< NUMENTITIES){
	vector3 accel_sum={0,0,0};
	accel_sum[k]+=dev_accels[i * NUMENTITIES + j][k];
		
//compute the new velocity based on the acceleration and time interval
//compute the new position based on the velocity and time interval
	dev_hVel[i][k]+=accel_sum[k]*INTERVAL;
	dev_hPos[i][k]+=dev_hVel[i][k]*INTERVAL;
	}
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	//vector3* dev_values;
	// vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	// cudaMalloc(&dev_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	// cudaMemcpy(dev_values, values,sizeof(vector3) * NUMENTITIES * NUMENTITIES,cudaMemcpyHostToDevice);

	int gridD = (NUMENTITIES/256) +1;
	// vector3** dev_accels;
	// cudaMalloc(&dev_accels, sizeof(vector3*) * NUMENTITIES);
	vector3* dev_accels;
	cudaMalloc(&dev_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

	

	// contructAccels<<<dimGrid, dimAc>>>(dev_accels, dev_values);

	double * dev_mass;
	vector3* dev_hPos;
	vector3* dev_hVel;
	cudaMalloc(&dev_mass, sizeof(double) * NUMENTITIES );
	cudaMemcpy(dev_mass, mass, sizeof(double) * NUMENTITIES,cudaMemcpyHostToDevice);
	cudaMalloc(&dev_hPos, sizeof(vector3) * NUMENTITIES );
	cudaMemcpy(dev_hPos, hPos,sizeof(vector3) * NUMENTITIES,cudaMemcpyHostToDevice);
	cudaMalloc(&dev_hVel, sizeof(vector3) * NUMENTITIES );
	cudaMemcpy(dev_hVel, hVel,sizeof(vector3) * NUMENTITIES,cudaMemcpyHostToDevice);
	
	//dim3 numBlocks((NUMENTITIES+15)/16, (NUMENTITIES+15)/16);
	dim3 blockSize(32, 32);
	int blockDimX = 32;
	int blockDimY = 32;

	int gridDim = (NUMENTITIES + blockDimX - 1) / blockDimX; 
	dim3 grid(gridDim, gridDim);
	accelComputeKernal<<<gridDim, blockSize>>>(dev_accels, dev_mass, dev_hPos);
	cudaDeviceSynchronize();
	// vector3* accels = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	// cudaMemcpy(accels, dev_accels, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	// for (int i = 0; i < NUMENTITIES; ++i) {
	// 	for (int j = 0; j < NUMENTITIES; ++j) {
	// 		printf("(%f, %f, %f) ", accels[i * NUMENTITIES + j][0], accels[i * NUMENTITIES + j][1], accels[i * NUMENTITIES + j][2]);
	// 	}
	// 	printf("\n");
	// }
	dim3 sblockSize(16, 16, 3);
	int sblockDimX = 16;
	int sblockDimY = 16;

	int sgridDim = (NUMENTITIES + sblockDimX - 1) / sblockDimX; 	
	sumRows<<<sgridDim, sblockSize>>>(dev_accels, dev_hPos, dev_hVel);
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(cudaError));
	}
	cudaMemcpy(hVel, dev_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, dev_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	
	//free(accels);
	cudaFree(dev_mass);
	cudaFree(dev_accels);
	cudaFree(dev_hPos);
	cudaFree(dev_hVel);
}
