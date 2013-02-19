extern "C" {
#include "../include/gpucalc.h"
}

#define NUMSTREAMS 16

//computes the sum of the elements of the size 32 sumVector, storing the result in sumVector[0] in 5 cycles
__device__ void reduceSum(float* sumVector, unsigned int tid, char startN)
{
	char n = startN >> 1;

	while(n > 0)
	{
		if(tid < n)
			sumVector[tid] = sumVector[tid] + sumVector[tid + n];
		n = n >> 1;//n /= 2
	}
}	

__device__ void calcHH(	float matelem,//the block containing a column to calculate the householder vector of
			unsigned int tid,//the tid of the calling thread
			float hhVector[],//the array to store the resulting vector in
			int k)//the step at which this was called
{
	float tmp;
	int sign;

	//read vectors in from below the diagonal(k)
	if(tid >= k)
		hhVector[tid] = matelem;
	else if(tid < k)//zero above diagonal
		hhVector[tid] = 0.0;

	tmp = hhVector[tid];
	//calculate sign(only used by kth thread
	sign = tmp >= 0 ? 1 : -1;
	
	//square each element
	hhVector[tid] *= hhVector[tid];
	
	//ideally only do required computation here; not necessarily 16, 8, 4, 2, 1
	//reduction to calculate the sum of squares of the 32 element vector
	reduceSum(hhVector, tid, 32);

	if(tid == k)//in kth element
	{
		//calculate sign*norm and put in kth element
		hhVector[k] = sign * __fsqrt_rn(hhVector[0]);

		//if norm not 0
		if(hhVector[k] != 0.0)
		{
			//add element in block at (k,k) to norm and store in vector
			hhVector[k] = tmp + hhVector[k];

			//divide because want v(k+1:m) = v(k+1:m)/v(k) = v(k+1:m) * (1/v(k))
			hhVector[k] = 1.0/hhVector[k];
		}
		else//if norm is zero, 
			hhVector[k] = 1.0;
	}

	// use reduceVector[0] to compute the local value multiplied by 1/reduceVector[0]
	if(tid > k)
		hhVector[tid] = tmp * hhVector[k];
	else if(tid < k)
		hhVector[tid] = 0.0;//zero above diagonal

	if(tid == k)//apart from element 0
		hhVector[k] = 1.0;//which is always 1
}

__device__ void applyHH(float blockRow[],// the 32*32 matrix block to compute over
			unsigned int tid,//the tid of the thread
			float workingVector[],//SHARED a working space of size 32 used in the computation
			int k,//the step at which the function was called, A(k:m, k:n) is submatrix to operate on
			float hhVectorelem)//element of vector storing the householder vector, zero above diagonal
{
	float 	y,//y will be -2 divided by the sum of squares of vector
		z;//z will be v[tid] * sum(A(:,j) .* v) * y

	int 	j, blockref;//column counter, reference in block

	//thread tid starts at (tid, k)
	blockref = (k*32) + tid;
	
	//read data for summation
	workingVector[tid] = hhVectorelem;
	//square elements of working vector
	workingVector[tid] *= workingVector[tid];

	//reduction to find sum of squares of workingVector
	reduceSum(workingVector, tid, 32);

	//make y a register equal to -2/sum of squares of v(v'*v)
	y = (-2.0) / workingVector[0];
	
	for(j = k; j < 32; j ++)//submatrix A(k:m,k:n)
	{
		//fill workingVector with the componentwise multiplication of householder vector (zero above k) and column j of block
		//workingVector[tid] = matblock[blockref] * hhVectorelem;
		workingVector[tid] = blockRow[j] * hhVectorelem;

		//reduction to find sum of multiplication of column of block with hhVector
		reduceSum(workingVector, tid, 32);
		
		//set z = tidth element of v times -2/v'v
		z = hhVectorelem * y;

		//multiply z by sum of componentwise multiplication of column with hhVector in workingVector[0]
		z *= workingVector[0];
		
		//add z to block(tid,j). zero above diagonal
		blockRow[j] += z;

		//row major storage, next column is 32 elements ahead
		blockref += 32;
	}
	
	if(tid > k)//store essential part of vector below diagonal
		blockRow[k] = hhVectorelem;//insert essential part of vector below diagonal in column k of block
}



__device__ void calcDoubleHH	(float topElem,//element of top block
				float lowElem,
				unsigned int tid,//id of calling thread
				float hhVector[], //SHARED 32x1 array to insert vector
				int k)//step at which called. use column k
{
	float tmp;//elemK not used in tid > 0
	int sign;

	if(tid < k)//zero above diagonal in top
		hhVector[tid] = 0.0;

	if(tid == k)//top nonzero element
		hhVector[k] = topElem;

	if(tid > k)//zero below diagonal in top
		hhVector[tid] = 0.0;

	//all read lower block in
	hhVector[tid + 32] = lowElem;

	if(tid == k)//kth thread holds non zero element in top block
	{
		sign = topElem >= 0 ? 1 : -1;
	}

	//all threads hold elements from bottom block
	tmp = hhVector[tid + 32];

	//square top nonzero in hhVector
	if(tid == k)
		hhVector[k] *= hhVector[k];

	//square each element in bottom block
	hhVector[tid + 32] *= hhVector[tid + 32];

	//reduce to compute sum of squares in 0th element of 64 element hhVector
	reduceSum(hhVector, tid, 64);

	if(tid == k)
	{
		//store sign * norm in kth position
		hhVector[k] = sign * __fsqrt_rn(hhVector[0]);

		if(hhVector[k] != 0.0)
		{
			//add sign*norm to kth element and store
			hhVector[k] = topElem + hhVector[k];
			
			//divide because want to divide by hhVector[k]
			hhVector[k] = 1.0/hhVector[k];
		}
		else//norm zero
			hhVector[k] = 1.0;
	}

	//normalise by multiplying by kth element
	if(tid != k)
		hhVector[32 + tid] = tmp * hhVector[k];
	if(tid == k)
		hhVector[32 + k] = tmp * hhVector[k];

	if(tid == k)//top part is 1 on diagonal
		hhVector[k] = 1.0;
	else if(tid != k)//zero elsewhere
		hhVector[tid] = 0.0;
}

__device__ void applyDoubleHH	(float topRow[],
				float lowRow[],
				unsigned int tid,
				float workingVector[],
				int k,
				float hhVectorElem)
{
	float	y,//-2/v'v
		zupp, zlow;//y * v[i] *sum(A(:,j) .* v) for both blocks

	int 	j;//column counter

	//copy hhVector and square square each element for summation
	if(tid == k)
		workingVector[tid] = 1.0;
	if(tid != k)
		workingVector[tid] = 0.0;

	workingVector[tid + 32] = hhVectorElem * hhVectorElem;
	
	//reduce to find sum of squares
	reduceSum(workingVector, tid, 64);
	
	//set y = -2/sum of squares
	y = (-2) / workingVector[0];

	//for each column
	for(j = k; j < 32; j ++)
	{
		//fill working vector[i] with top block(i,j) * hhVector[i]
		if(tid == k)
			workingVector[tid] = topRow[j];
		if(tid != k)
			workingVector[tid] = 0.0;

		//fill workingVector[i + 32] with bottom block(i,j) * hhVector[i+32]
		workingVector[tid + 32] = lowRow[j] * hhVectorElem;

		//sum to find sum of componentwise multiplication
		reduceSum(workingVector, tid, 64);
		
		//set zupp = tidth element of hhvector times -2/v'v
		if(tid == k)
			zupp = y;
		if(tid != k)
			zupp = 0.0;

		zlow = y * hhVectorElem;

		//multiply both by sum of multiplication
		zupp *= workingVector[0];
		zlow *= workingVector[0];
		
		//add to top block element
		topRow[j] += zupp;

		//add to bottom block element
		lowRow[j] += zlow;
	}
	
	lowRow[k] = hhVectorElem;
}


__device__ void applyHHPrime	(float velem,
				unsigned int tid,
				float applyRow[],
				int k,
				float workingVector[])
{
	//apply the vector starting at row k+1 of hhVector to the (k:32) portion of applyVector
	float y, z;
	
	int j;

	//load workingVector with squares of hhVector
	workingVector[tid] = velem * velem;

	//sum to find sum of squares of hhVector
	reduceSum(workingVector, tid, 32);

	//set y = -2/sumsquares
	y = (-2) / workingVector[0];

	//apply to columns 0:32
	for(j = 0; j < 32; j ++)
	{
		//load working vector with componentwise multiplication of hhVector with column of block A
		workingVector[tid] = velem * applyRow[j];

		//sum to find sum of pairwise mult
		reduceSum(workingVector, tid, 32);

		//set z = sum of this times -2/sumvector
		z = y * velem;

		//multiply by element of hhVector
		z *= workingVector[0];
	
		//add to applyVector in place
		applyRow[j] += z;
	}
}



__device__ void applyDoubleHHPrime	(float velem,
					unsigned int tid,
					float topAppRow[],
					float lowAppRow[],
					int k,
					float workingVector[])
{
	float y, z;

	int j;

	//load implied top vector
	if(tid == k)
		workingVector[k] = 1.0;
	
	if(tid != k)
		workingVector[tid] = 0.0;
	
	//square lower elements (top is 001000 so no need to multiply)
	workingVector[tid + 32] = velem * velem;

	//reduce to find sum of squares
	reduceSum(workingVector, tid, 64);

	//set y = -2/sum of squares
	y = (-2) / workingVector[0];

	//apply to columns (0:32)
	for(j = 0; j < 32; j ++)
	{
		//load workingVector with 1 * *topAppElem (want workingVector to have component mult of hhvector and elements of application block
		if(tid == k) 
			workingVector[k] = topAppRow[j];

		if(tid != k)
			workingVector[tid] = 0.0;
	
		//do the same for the lower block but nonzero here
		workingVector[tid + 32] = velem * lowAppRow[j];

		//reduce to find sum of multiplication
		reduceSum(workingVector, tid, 64);

		//set zTop and zLow equal to this sum times -2/v'v
		z = workingVector[0] * y;
	
		//multiply top by sum(mult) * -2/v'v. (velem here "is" 1)
		if(tid == k)
			topAppRow[j] += z;

		//multiply zLow by lower element of hhVector
		z *= velem;

		//add result to lower application element
		lowAppRow[j] += z;
	}
}

__global__ void doQRS(float* matrix, int ldm)
{
	__shared__ float workingVector[32];

	unsigned int tid = threadIdx.x;

	int k, j;

	float bRow[32];

	for(j = 0; j < 32; j ++)//load row of block into local
		bRow[j] = matrix[(j*ldm) + tid];

	for(k = 0; k < 32; k ++)
	{
		//calculate the kth hh vector from the kth column of the tidth row of the matrix
		calcHH(bRow[k], tid, workingVector, k);

		//calculate the application of the hhvector along row tid
		applyHH(bRow, tid, workingVector, k, workingVector[tid]);
	}

	//copy row back
	for(j = 0; j < 32; j ++)
		matrix[(j*ldm) + tid] = bRow[j];
}

__global__ void doQRD(float* blockA, float* blockB, int ldm)
{
	__shared__ float workingVector[64];

	unsigned int tid = threadIdx.x;

	int k, j;

	float topRow[32];
	float lowRow[32];
	
	for(j = 0; j < 32; j ++)//for each column
	{
		//read top block
		topRow[j] = blockA[(j*ldm) + tid];

		//read lower block into lower 32x32 square
		lowRow[j] = blockB[(j*ldm) + tid];
	}

	for(k = 0; k < 32; k ++)
	{
		//calculate and store the vector
		calcDoubleHH	(topRow[k],
				lowRow[k],
				tid,
				workingVector,
				k);

		//apply vector to both tidth rows of the matrix
		applyDoubleHH	(topRow,
				lowRow,
				tid,
				workingVector,
				k,
				workingVector[tid + 32]);
	}

	for(j = 0; j < 32; j ++)
	{
		//write back to correct blocks
		blockA[(j*ldm) + tid] = topRow[j];
		blockB[(j*ldm) + tid] = lowRow[j];
	}
}

__global__ void doSAPP(float* blockV, float* blockA, int ldm)
{
	__shared__ float workingVector[32];

	unsigned int tid = threadIdx.x;
	float hhVelems[32];
	float applyRow[32];

	int j, k;

	for(j = 0; j < 32; j ++)//load tidth row of hhvectors
	{
		if(tid == j) 
			hhVelems[j] = 1.0;
		if(tid < j)
			hhVelems[j] = 0.0;
		if(tid > j)
			hhVelems[j] = blockV[(j*ldm) + tid];

		//load row to apply
		applyRow[j] = blockA[(j*ldm) + tid];
	}

	for(k = 0; k < 32; k ++)
	{
		
		//apply kth vector to columns (1:32) of A

		applyHHPrime	(hhVelems[k],//tidth element of kth vector
				tid,
				applyRow,
				k,
				workingVector);
	}
		
	for(j = 0; j < 32; j ++)
		blockA[(j*ldm) + tid] = applyRow[j];
}

__global__ void doDAPP(float* blockV, float* blockA, float* blockB, int ldm)
{
	__shared__ float workingVector[64];

	unsigned int tid = threadIdx.x;

	int j, k;

	float vElems[32];
	float topApplyRow[32];
	float lowApplyRow[32];

	for(j = 0; j < 32; j ++)//load vElems with bottom hh vector row
	{
		vElems[j] = blockV[(j*ldm) + tid];

		//load rows to apply with data
		topApplyRow[j] = blockA[(j*ldm) + tid];
		lowApplyRow[j] = blockB[(j*ldm) + tid];
	}

	//apply vector k to rows k:32 of columns 0:32 of A and B
	for(k = 0; k < 32; k ++)
	{
		applyDoubleHHPrime	(vElems[k],
					tid,
					topApplyRow,
					lowApplyRow,
					k,
					workingVector);
	}

	for(j = 0; j < 32; j ++)
	{
		blockA[(j*ldm) + tid] = topApplyRow[j];
		blockB[(j*ldm) + tid] = lowApplyRow[j];
	}
}

extern "C"
void cudaQRFull(float* mat, int m, int n)
{
	int i, j, k, p, q, s;
	int blockdm;

	float* dev_m, *dev_K, *dev_V, *dev_A, *dev_B;

	cudaStream_t streams[NUMSTREAMS];
	
	for(k = 0; k < NUMSTREAMS; k ++)
		cudaStreamCreate(&streams[k]);

	p = m/32;
	q = n/32;

	blockdm = 32*m;//block to block dim along row

	cudaMalloc((void**) &dev_m, m*n*sizeof(float));
	cudaMemcpy(dev_m, mat, m*n*sizeof(float), cudaMemcpyHostToDevice);

	dev_K = dev_m;

	for(k = 0; k < q; k ++)
	{
		doQRS<<<1, 32, 0, streams[0]>>>(dev_K, m);
		cudaDeviceSynchronize();

		s = 1;

		dev_A = dev_K + blockdm;//one along
		for(j = k+1; j < q; j ++)
		{
			doSAPP<<<1, 32, 0, streams[s]>>>(dev_K, dev_A, m);
			
			dev_A += blockdm;//advance along row

			s ++;
			s = s % (NUMSTREAMS - 1);
		}

		dev_V = dev_K + 32;//one down from K

		for(i = k+1; i < p; i ++)
		{
			doQRD<<<1, 32, 0, streams[0]>>>(dev_K, dev_V, m);
			cudaDeviceSynchronize();

			s = 0;

			dev_A = dev_K + blockdm;//one along from K			
			dev_B = dev_V + blockdm;//one along from V

			for(j = k+1; j < q; j ++)
			{
				doDAPP<<<1, 32, 0, streams[s]>>>(dev_V, dev_A, dev_B, m);
				
				dev_A += blockdm;
				dev_B += blockdm;
				
				s ++;
				s = s % NUMSTREAMS;
			}
			dev_V += 32;//one down from previous
		}
		dev_K += blockdm + 32;//one along, one down
		cudaDeviceSynchronize();
	}

	cudaMemcpy(mat, dev_m, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_m);
	for(k = 0; k < NUMSTREAMS; k ++)
		cudaStreamDestroy(streams[k]);
}

extern "C"
void cudaDAPP(float* vBlock, float* aBlock, float* bBlock, int ldm)
{
	float* dev_bV, *dev_bA, *dev_bB;
	int j, lddev = 96;

	cudaMalloc((void**) &dev_bV, lddev*32*sizeof(float));

	//v on top of a on top of b in 96x32 array on gpu
	dev_bA = dev_bV + 32;
	dev_bB = dev_bA + 32;

	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(dev_bV + (lddev*j), vBlock + (ldm*j), 32*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bA + (lddev*j), aBlock + (ldm*j), 32*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bB + (lddev*j), bBlock + (ldm*j), 32*sizeof(float), cudaMemcpyHostToDevice);
	}

	doDAPP<<<1, 32>>>(dev_bV, dev_bA, dev_bB, lddev);

	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(aBlock + (ldm*j), dev_bA + (lddev*j), 32*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bBlock + (ldm*j), dev_bB + (lddev*j), 32*sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	cudaFree(dev_bV);
}

extern "C"
void cudaSAPP(float* vBlock, float* block, int ldm)
{
	float* dev_bV, *dev_bB;
	int j;
	
	//allocate 32x64 blocks(2)
	cudaMalloc((void**) &dev_bV, 32*64*sizeof(float));

	//start of second block row major
	dev_bB = dev_bV + (32*32);

	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(dev_bV + (32*j), vBlock + (ldm*j), 32*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bB + (32*j), block + (ldm*j), 32*sizeof(float), cudaMemcpyHostToDevice);
	}
	
	doSAPP<<<1,32>>>(dev_bV, dev_bB, 32);

	
	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(block + (ldm*j), dev_bB + (32*j), 32*sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaFree(dev_bV);
}
	

extern "C"
void cudaQRS(float* block, int ldm)
{
	float* dev_b;
	int j;

	cudaMalloc((void**) &dev_b, 32*32*sizeof(float));

	//transfer 32x32 block over
	for(j = 0; j < 32; j ++)
		cudaMemcpy(dev_b + (j*32), block + (j*ldm), 32*sizeof(float), cudaMemcpyHostToDevice);

	doQRS<<<1, 32>>>(dev_b, 32);

	for(j = 0; j < 32; j ++)
		cudaMemcpy(block + (j*ldm), dev_b + (j*32), 32*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_b);
}

extern "C"
void cudaQRD(float* blockA, float* blockB, int ldm)
{
	float* dev_bA, *dev_bB;
	int j, lddev = 64;

	cudaMalloc((void**) &dev_bA, lddev*32*sizeof(float));

	dev_bB = dev_bA + lddev - 32;

	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(dev_bA + (j*lddev), blockA + (j*ldm), 32*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bB + (j*lddev), blockB + (j*ldm), 32*sizeof(float), cudaMemcpyHostToDevice);
	}

	doQRD<<<1, 32>>>(dev_bA, dev_bB, lddev);

	for(j = 0; j < 32; j ++)
	{
		cudaMemcpy(blockA + (j*ldm), dev_bA + (j*lddev), 32*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(blockB + (j*ldm), dev_bB + (j*lddev), 32*sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	cudaFree(dev_bA);
}

