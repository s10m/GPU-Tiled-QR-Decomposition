#ifndef GPUCOMP_H
#define GPUCOMP_H
void cudaQRS(float*, int);
void cudaSAPP(float*, float*, int);
void cudaQRD(float*, float*, int);
void cudaDAPP(float*, float*, float*, int);

void cudaQRFull(float*, int, int);
void docudastuff(float*, float*, int);
#endif
