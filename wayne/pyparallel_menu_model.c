#include "pyparallel_menu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define PI 3.14159265358979323846


double pfibo(int n) {
 int i;
 double a=0.0, b=1.0, tmp;
 for (i=0; i<n; ++i) {
 tmp = a; a = a + b; b = tmp;
 }
 return a;
}

double sump(double *array,int N) {
 double ssum;
 int i,j;
 ssum = 0.0;
 #pragma omp parallel num_threads(xxxxxx) \
 			shared(array) \
 			private(i,j) \
 			firstprivate(N)\
 			reduction(+:ssum)
 {

 #pragma omp for schedule(static)	 			
 for (i=0;i<N;i++){
  for (j=0;j<N;j++) {
   ssum += array[i];
  }
 }

 }

 return ssum;
}





/* Box-Muller parallel - for generating a normal distribution*/
double *Box_Muller_parallel(double mu,double sigma,int size) {
	
	int i,N=size/2;
	unsigned seed;
	double *Xarr = malloc(size*sizeof(double));
	double R,theta;
	
	
	#pragma omp parallel num_threads(xxxxxx)\
				shared(Xarr)\
				private(i,seed,R,theta)\
				firstprivate(N)
	 
	{
			
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = myid*N/thread_number , iend = (myid+1)*N/thread_number;
	if (myid == thread_number -1) {iend = N;}
	
	
	seed = 25234 + 17*myid + time(NULL);
	
		
	for (i=istart;i<iend;i++) {
				
		theta = 2.*PI*rand_r(&seed)/((double)RAND_MAX);
		R = sqrt(-2.*log(rand_r(&seed)/((double)RAND_MAX)));
		
		Xarr[i] = (R *cos(theta) )*sigma + mu;
		Xarr[i+N] = (R *sin(theta) )*sigma + mu;
	}
	
	}
	
	return Xarr;
}



/* Box-Muller parallel - for generating a normal distribution*/
double *Box_Muller_parallel_new(double mu,double *sigma,int size) {
	
	int i,N=size/2;
	unsigned seed;
	double *Xarr = malloc(size*sizeof(double));
	double R,theta;
	
	
	#pragma omp parallel num_threads(xxxxxx)\
				shared(Xarr,sigma,mu)\
				private(i,seed,R,theta)\
				firstprivate(N)
	 
	{
			
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = myid*N/thread_number , iend = (myid+1)*N/thread_number;
	if (myid == thread_number -1) {iend = N;}
	
	
	seed = 25234 + 17*myid + time(NULL);
	
		
	for (i=istart;i<iend;i++) {
				
		theta = 2.*PI*rand_r(&seed)/((double)RAND_MAX);
		R = sqrt(-2.*log(rand_r(&seed)/((double)RAND_MAX)));
		
		Xarr[i] = (R *cos(theta) )*sigma[i] + mu;
		Xarr[i+N] = (R *sin(theta) )*sigma[i+N] + mu;
	}
	
	}
	
	return Xarr;
}

int *Batman(int *counts,int size,double  *x_pos,double  *y_pos, double *psf_ratio, double *psf_sigmal,double *psf_sigmah,int nr,int nc, int test){
	
	int i,k,ssum,dim=nr*nc,N;
	int ii,jj,electron_index,electron_counter,xpos,ypos;
	double R,theta;
	unsigned seed;
	int *pixel_array = malloc(dim*sizeof(int));
	
	
	ssum = 0;
	#pragma omp parallel num_threads(xxxxxx)\
				shared(counts)\
				private(k)\
				firstprivate(size)\
				reduction(+:ssum)
	{
	
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = (myid*size/thread_number) , iend = ((myid+1)*size/thread_number);
	if (myid == thread_number -1) {iend = size;}
	
	for (k=istart;k<iend;k++){
		ssum += counts[k];
		}
	}
	
	double *A = malloc((2*ssum)*sizeof(double));
	
	
	
	#pragma omp parallel num_threads(xxxxxx)\
				shared(A)\
				private(i,seed,R,theta)\
				firstprivate(ssum)
	 
	{
			
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = myid*ssum/thread_number , iend = (myid+1)*ssum/thread_number;
	if (myid == thread_number -1) {iend = ssum;}
	
	
	seed = 25234 + 17*myid + time(NULL) + test;
	
		
	for (i=istart;i<iend;i++) {
				
		theta = 2.*PI*rand_r(&seed)/((double)RAND_MAX);
		R = sqrt(-2.*log(rand_r(&seed)/((double)RAND_MAX)));
		
		A[i] = R *cos(theta) ;
		A[i+ssum] = R *sin(theta) ;
	}
	
	}
	
	
	/*--------- filling pixe_array with zeros--------------*/
	#pragma omp parallel num_threads(xxxxxx)\
				shared(pixel_array)\
				private(i)\
				firstprivate(dim)
	 
	{
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = myid*dim/thread_number , iend = (myid+1)*dim/thread_number;
	if (myid == thread_number -1) {iend = dim;}
	
	for (i=istart;i<iend;i++) {
				
		pixel_array[i] = 0;
	}
		
	}
	/*--------- filling pixel_array with zeros--------------*/


	electron_counter = 0;
	for (ii=0;ii<size;ii++){
		N = counts[ii]*psf_ratio[ii] ;
		for (jj=0;jj<N;jj++){
			xpos = A[electron_counter]*psf_sigmah[ii] + x_pos[ii];
			ypos = A[electron_counter+ssum]*psf_sigmah[ii] + y_pos[ii];
			if ( xpos>0 && xpos<nr && ypos>0 && ypos<nc) {
				electron_index = ypos*nc + xpos;
				pixel_array[electron_index] = pixel_array[electron_index] + 1;
			}
			electron_counter = electron_counter + 1;
		}
		for (jj=0;jj<(counts[ii]-N);jj++) {
			xpos = A[electron_counter]*psf_sigmal[ii] + x_pos[ii];
			ypos = A[electron_counter+ssum]*psf_sigmal[ii] + y_pos[ii];
			if ( xpos>0 && xpos<nr && ypos>0 && ypos<nc) {
				electron_index = ypos*nc + xpos;
				pixel_array[electron_index] = pixel_array[electron_index] + 1;
			}
			electron_counter = electron_counter + 1;
		}
	}
	
	
	free(A);
	return pixel_array;
}




/*double *feval(double *array,int size,double (*func)(double)) {
 int i;
 double *outarray;

 outarray = malloc(size*sizeof(double));
 for (i=0;i<size;i++) {
  outarray[i] = func(array[i]);
 }

 return outarray; 
}*/
