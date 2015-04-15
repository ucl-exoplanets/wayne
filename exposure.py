""" An exposure object, this is designed to be a fits file like object, taking input from a generation function and able
to construct the output along with certain visualisation methods.
"""

import astropy.io.fits as fits


class Exposure(object):

    def __init__(self):
        """ Sets up the exposure class. We probably need to give the class some information
        :return:
        """

        self.reads = []  # read 0 ->

    def add_read(self, data):
        """ adds the read to the exposure, will probably need some header information to.

        :param data:
        :return:
        """

        self.reads.append(data)

    def generate_fits(self, out_path):
        """ Saves the exposure as a fits file.
        :return:
        """


        science_header = fits.PrimaryHDU()  # Lots of quantities for this will be defined elsewhere

        hdulist = fits.HDUList([science_header])

        for i, data in enumerate(reversed(self.reads)):

            read_HDU = fits.ImageHDU(data)
            error_array = fits.ImageHDU()

            """ This array contains 16 independent flags indicating various status and problem conditions associated
            with each corresponding pixel in the science image. Each flag has a true (set) or false (unset) state and is
            encoded as a bit in a 16-bit integer word. Users are advised that this word should not be interpreted as a
            simple integer, but must be converted to base-2 and each bit interpreted as a flag. Table 2.5 lists the WFC3
            data quality flags.
            """
            data_quality_array = fits.ImageHDU()

            """ This array is present only for IR data. It is a 16-bit integer array and contains the number of samples
            used to derive the corresponding pixel values in the science image. For raw and intermediate data files, the
            sample values are set to the number of readouts that contributed to the science image. For calibrated files,
            the SAMP array contains the total number of valid samples used to compute the final science image pixel
            value, obtained by combining the data from all the readouts and rejecting cosmic ray hits and saturated
            pixels. Similarly, when multiple exposures (i.e., REPEAT-OBS) are combined to produce a single image,
            the SAMP array contains the total number of samples retained at each pixel for all the exposures.
            """
            samples_HDU = fits.ImageHDU()

            """ This array is present only for IR data. This is a floating-point array that contains the effective
            integration time associated with each corresponding science image pixel value. For raw and intermediate
            data files, the time value is the total integration time of data that contributed to the science image.
            For calibrated datasets, the TIME array contains the combined exposure time of the valid readouts or
            exposures that were used to compute the final science image pixel value, after rejection of cosmic
            rays and saturated pixels from the intermediate data.
            """
            integration_time_HDU = fits.ImageHDU()

            hdulist.extend([read_HDU, error_array, data_quality_array, samples_HDU, integration_time_HDU])

        hdulist.writeto(out_path)