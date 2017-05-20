import numpy as np
cimport numpy as np


cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)

cdef extern from "pyparallel_menu.h":
    cdef int *PSF(int *counts, int size, double  *x_pos, double  *y_pos,
                  double *psf_ratio, double *psf_sigmal, double *psf_sigmah,
                  int nr, int nc, int test, int threads) nogil

def apply_psf(np.ndarray counts, np.ndarray pos_x, np.ndarray pos_y,
              np.ndarray ratio_psf, np.ndarray sigmal_psf,
              np.ndarray sigmah_psf, int NR, int NC, int test, int threads):
    cdef:
        int counts_size
        int j
        int *carray

    counts_size = len(counts)
    cdef int *ccounts = <int*> malloc(counts_size * sizeof(int))
    for j in range(counts_size):
        ccounts[j] = counts[j]

    carray = PSF(ccounts, counts_size, (<double *> pos_x.data),
                 (<double *> pos_y.data), (<double *> ratio_psf.data),
                 (<double *> sigmal_psf.data), (<double *> sigmah_psf.data),
                 NR, NC, test, threads)
    pyarray = np.zeros(NR * NC)

    for j in range(NR * NC):
        pyarray[j] = carray[j]

    free(carray)
    free(ccounts)
    return pyarray
