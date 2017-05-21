#include "pyparallel_menu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define PI 3.14159265358979323846

int *PSF(int *counts,int size,double  *x_pos,double  *y_pos, double *psf_ratio, double *psf_sigmal,double *psf_sigmah,int nr,int nc, int test, int threads){
	
	int i,k,ssum,dim=nr*nc,N;
	int ii,jj,electron_index,electron_counter,xpos,ypos;
	double R,theta;
	unsigned seed;
	int *pixel_array = malloc(dim*sizeof(int));
	
	
	ssum = 0;
	#pragma omp parallel num_threads(threads)\
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
	
	
	
	#pragma omp parallel num_threads(threads)\
				shared(A)\
				private(i,seed,R,theta)\
				firstprivate(ssum)
	 
	{
			
	int myid = omp_get_thread_num(),thread_number = omp_get_num_threads();
	int istart = myid*ssum/thread_number , iend = (myid+1)*ssum/thread_number;
	if (myid == thread_number -1) {iend = ssum;}
	
	
	seed = 25234 + 17*myid + test;
	
		
	for (i=istart;i<iend;i++) {
				
		theta = 2.*PI*rand_r(&seed)/((double)RAND_MAX);
		R = sqrt(-2.*log(rand_r(&seed)/((double)RAND_MAX)));
		
		A[i] = R *cos(theta) ;
		A[i+ssum] = R *sin(theta) ;
	}
	
	}
	
	
	/*--------- filling pixe_array with zeros--------------*/
	#pragma omp parallel num_threads(threads)\
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
