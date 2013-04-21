#ifndef GPUCOMP_H
#define GPUCOMP_H
void cudaQRTask(float*, int, int, int, int);
void cudaQRFull(float*, int, int);
void testDAPP(float*, int, int);
void doCUDADAPP(float*);
#endif
