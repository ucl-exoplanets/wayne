import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cimport cpython


cdef extern from "stdlib.h":
	void *malloc(size_t size)
	void free(void *ptr)

cdef extern from "pyparallel_menu.h":
	cdef double pfibo(int n) nogil
	cdef double sump(double *array,int size) nogil
	cdef double *Box_Muller_parallel(double mu,double sigma,int size) nogil
	cdef double *Box_Muller_parallel_new(double mu,double *sigma,int size) nogil
	cdef int *Batman(int *counts,int size,double  *x_pos,double  *y_pos, double *psf_ratio, double *psf_sigmal,double *psf_sigmah,int nr,int nc, int test) nogil
	#double *feval(double *array,int size,double (*func)(double)) 

def fibop(int n):
	return pfibo(n)


#np.ndarray[double, ndim=1, mode="c"]
def psum(np.ndarray x):
	cdef:
		int x_size
	x_size = len(x)

	return sump(( <double *> x.data),x_size)

def paranormal(double m,double s,int N):
	cdef:
		int i
		double *dist
		#np.double_t[:] view = <np.double_t[:N]> dist
		np.ndarray[double, ndim=1 ,mode="c"] pyarray
		#cpython.PyObject *o	
	
	pyarray = np.zeros(N, dtype=np.double)
	dist = Box_Muller_parallel(m,s,N)
	#o = <cpython.PyObject *> dist
	#cpython.Py_XINCREF(o)
	for i in range(N):
		pyarray[i] = dist[i]

	return pyarray

def paranormal1(double mmu,np.ndarray ssigma):
	cdef:
		int ssigma_size
		int j
		np.ndarray[double, ndim=1 ,mode="c"] pyarray
		double *crandn
	ssigma_size = len(ssigma)
	pyarray = np.zeros(ssigma_size, dtype=np.double)
	crandn = Box_Muller_parallel_new(mmu,( <double *> ssigma.data),ssigma_size)

	for j in range(ssigma_size):
		pyarray[j] = crandn[j]

	free(crandn)
	return pyarray
	

def apply_psf(np.ndarray counts , np.ndarray pos_x,np.ndarray pos_y , np.ndarray ratio_psf , np.ndarray sigmal_psf , np.ndarray sigmah_psf,int NR , int NC, int test):

	cdef:
		int counts_size
		int j
		int *carray
		#double aa

	counts_size = len(counts)
	cdef int * ccounts = <int*>malloc(counts_size*sizeof(int))
	for j in range(counts_size):
		ccounts[j] = counts[j]




	carray = Batman( ccounts,counts_size,( <double *> pos_x.data), ( <double *> pos_y.data) , ( <double *> ratio_psf.data), ( <double *> sigmal_psf.data),( <double *> sigmah_psf.data), NR , NC, test)
	pyarray = np.zeros(NR*NC)
	

	for j in range(NR*NC):
		pyarray[j] =  carray[j]

	free(carray)
	free(ccounts)
	return pyarray
		
	





